"""Grouped perturbation branches for the toy attraction experiment.

The ordinary campaign runner executes one branch at a time.  This module keeps
the same scalar intervention equations while grouping equal-policy branches in
``ReplicaBank``.  A group has one atomic checkpoint containing every branch's
parameters, Adam state, sampler cursor, and lineage.  The public entry points
are deliberately callback-friendly because source checkpoints and evaluation
code live in the Colab campaign layer.

The runner is dependency-light at import time.  PyTorch is imported only when a
branch is materialised or continued, which keeps local schema and lineage
checks cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, TypeAlias

from .batched_training import BatchedTrainingState, ReplicaBank
from .checkpoints import CheckpointError, CheckpointStore
from .episodes import collate
from .evaluation import EvaluationMetrics, ModeThresholds, classify_mode, evaluate_agent
from .perturbations import classify_recovery, make_lineage
from .rewards import RewardConfig
from .types import Mode, PairedEpisode


PERTURBATION_BANK_SCHEMA_VERSION = 1
TABLE_SCHEMA_VERSION = 1
DEFAULT_HORIZONS = (0, 2_000, 10_000, 40_000)
DEFAULT_INTERVENTIONS = (
    ("gaussian_noise", 0.01),
    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.10),
    ("off_midpoint", 0.25),
    ("off_midpoint", 0.50),
    ("off_midpoint", 0.75),
    ("gate_attenuation", 0.50),
    ("gate_attenuation", 0.00),
    ("opposite_pulse", 50.0),
)


MetricLike: TypeAlias = EvaluationMetrics | Mapping[str, object] | Sequence[object]
ModelFactory: TypeAlias = Callable[..., object]
MetricEvaluator: TypeAlias = Callable[[object, Sequence[PairedEpisode]], MetricLike]
BankMetricEvaluator: TypeAlias = Callable[..., object]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _jsonable(value: object) -> object:
    """Convert metadata to deterministic JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # A killed worker may leave one partial tail.  Committed rows are
            # retained and the next callback rewrites a clean prefix.
            break
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = "".join(
        json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping_digest(value: Mapping[str, object]) -> str:
    """Hash a small metadata mapping such as a checkpoint RNG payload."""

    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return _hash_bytes(payload.encode("utf-8"))


def _tensor_hash(value: object) -> str:
    """Hash a scalar model state without importing torch at module load."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "contiguous"):
        value = value.contiguous()
    if hasattr(value, "numpy"):
        try:
            array = value.numpy()
            return _hash_bytes(array.tobytes(order="C"))
        except Exception:
            pass
    return _hash_bytes(repr(_jsonable(value)).encode("utf-8"))


def state_fingerprint(state: Mapping[str, object]) -> str:
    """Return a deterministic digest of a model state mapping."""

    digest = hashlib.sha256()
    for name in sorted(str(key) for key in state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_hash(state[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _mapping_value(mapping: Mapping[str, object], *names: str, default: object = None) -> object:
    lowered = {str(key).lower(): key for key in mapping}
    for name in names:
        if name in mapping:
            return mapping[name]
        key = lowered.get(name.lower())
        if key is not None:
            return mapping[key]
    return default


def _metric_mapping(metric: MetricLike) -> dict[str, object]:
    if isinstance(metric, EvaluationMetrics):
        return {
            "c_on": float(metric.c_on),
            "c_off": float(metric.c_off),
            "gap": float(metric.gap or 0.0),
            "goal": None if metric.goal is None else float(metric.goal),
            "gate": None if metric.gate is None else float(metric.gate),
            "off_audit_logit": (
                None if metric.off_audit_logit is None else float(metric.off_audit_logit)
            ),
            "train_reward": (
                None if metric.train_reward is None else float(metric.train_reward)
            ),
            "label": metric.mode.value if metric.mode else Mode.INTERMEDIATE.value,
            "step": int(metric.checkpoint_step),
        }
    if isinstance(metric, Mapping):
        c_on = _mapping_value(metric, "c_on", "C_on", "on")
        c_off = _mapping_value(metric, "c_off", "C_off", "off")
        if c_on is None or c_off is None:
            raise KeyError("metric mapping must contain c_on/C_on and c_off/C_off")
        result = {str(key): _jsonable(value) for key, value in metric.items()}
        result["c_on"] = _finite(c_on, "c_on")
        result["c_off"] = _finite(c_off, "c_off")
        result["gap"] = result["c_on"] - result["c_off"]
        for name in ("goal", "gate", "off_audit_logit", "train_reward"):
            value = _mapping_value(metric, name)
            result[name] = None if value is None else _finite(value, name)
        return result
    values = tuple(metric)
    if len(values) < 2:
        raise ValueError("metric sequences need c_on and c_off")
    result = {
        "c_on": _finite(values[0], "c_on"),
        "c_off": _finite(values[1], "c_off"),
        "goal": None if len(values) < 3 else _finite(values[2], "goal"),
        "gate": None if len(values) < 4 else _finite(values[3], "gate"),
    }
    result["gap"] = result["c_on"] - result["c_off"]
    return result


def _metric_for_label(metric: Mapping[str, object], thresholds: ModeThresholds) -> str:
    goal = metric.get("goal")
    return classify_mode(
        c_on=float(metric["c_on"]),
        c_off=float(metric["c_off"]),
        goal=None if goal is None else float(goal),
        thresholds=thresholds,
    ).value


def _source_metric(source: Mapping[str, object]) -> dict[str, object]:
    candidate = _mapping_value(source, "source_metric", "metric", "metrics")
    if isinstance(candidate, Mapping) or (
        isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes))
    ):
        result = _metric_mapping(candidate)  # type: ignore[arg-type]
    else:
        result = _metric_mapping(source)
    if result.get("label") in (None, ""):
        result["label"] = str(
            _mapping_value(source, "source_mode", "label", default=Mode.INTERMEDIATE.value)
        )
    if result.get("step") in (None, ""):
        result["step"] = int(_mapping_value(source, "source_step", "step", default=0) or 0)
    return result


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A scalar source checkpoint plus its compact behavioral metadata."""

    source_run_id: str
    source_step: int
    source_mode: str
    source_metric: Mapping[str, object]
    model_state: Mapping[str, object]
    optimizer_state: Mapping[str, object]
    training_state: object
    rng_state: Mapping[str, object] = field(default_factory=dict)
    sampler_seed: int = 0
    model_seed: int = 0
    source_checkpoint_hash: str = ""
    source_rng_digest: str = ""
    source_config_sha256: str = ""
    source_archive_sha256: str = ""
    raw: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_run_id, str) or not self.source_run_id.strip():
            raise ValueError("source_run_id must be non-empty")
        if isinstance(self.source_step, bool) or int(self.source_step) < 0:
            raise ValueError("source_step must be non-negative")
        if not isinstance(self.source_mode, str) or not self.source_mode.strip():
            raise ValueError("source_mode must be non-empty")
        if not isinstance(self.model_state, Mapping) or not self.model_state:
            raise ValueError("model_state must be a non-empty mapping")
        if not isinstance(self.optimizer_state, Mapping):
            raise ValueError("optimizer_state must be a mapping")
        if not isinstance(self.rng_state, Mapping):
            raise ValueError("rng_state must be a mapping")
        object.__setattr__(self, "source_step", int(self.source_step))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SourceSnapshot":
        """Decode source records emitted by the campaign/checkpoint layer."""

        payload: Mapping[str, object] = value
        nested = _mapping_value(value, "torch_state", "checkpoint_payload", "payload")
        if isinstance(nested, Mapping):
            payload = nested
        model_state = _mapping_value(payload, "model_state", "state_dict")
        optimizer_state = _mapping_value(payload, "optimizer_state", default={})
        training_state = _mapping_value(payload, "training_state", "state")
        if not isinstance(model_state, Mapping):
            raise ValueError("source record has no model_state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("source record has no optimizer_state mapping")
        if training_state is None:
            training_state = {}
        payload_rng = _mapping_value(payload, "rng_state", "torch_rng", default={})
        outer_rng = _mapping_value(value, "rng_state", default={})
        rng: dict[str, object] = {}
        if isinstance(outer_rng, Mapping):
            rng.update(outer_rng)
        if isinstance(payload_rng, Mapping):
            rng.update(payload_rng)
        source_metric = _source_metric(value)
        source_step = int(
            _mapping_value(
                value,
                "source_step",
                "checkpoint_step",
                "step",
                default=source_metric.get("step", 0),
            )
            or 0
        )
        source_mode = str(
            _mapping_value(
                value,
                "source_mode",
                "source_label",
                "label",
                default=source_metric.get("label", Mode.INTERMEDIATE.value),
            )
        )
        source_metric["label"] = source_mode
        source_metric["step"] = source_step
        source_hash = str(
            _mapping_value(
                value,
                "source_checkpoint_hash",
                "checkpoint_hash",
                "metadata_sha256",
                default="",
            )
            or ""
        )
        rng_digest = str(
            _mapping_value(
                value,
                "source_rng_digest",
                "rng_digest",
                default=_mapping_digest(rng) if rng else "",
            )
            or ""
        )
        return cls(
            source_run_id=str(
                _mapping_value(value, "source_run_id", "run_id", default="source")
            ),
            source_step=source_step,
            source_mode=source_mode,
            source_metric=source_metric,
            model_state=dict(model_state),
            optimizer_state=dict(optimizer_state),
            training_state=training_state,
            rng_state=rng,
            sampler_seed=int(_mapping_value(value, "sampler_seed", "shuffle_seed", default=0) or 0),
            model_seed=int(_mapping_value(value, "model_seed", "seed", default=0) or 0),
            source_checkpoint_hash=source_hash,
            source_rng_digest=rng_digest,
            source_config_sha256=str(
                _mapping_value(value, "source_config_sha256", "config_sha256", default="")
                or ""
            ),
            source_archive_sha256=str(
                _mapping_value(value, "source_archive_sha256", "source_identity", default="")
                or ""
            ),
            raw=dict(value),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_run_id": self.source_run_id,
            "source_step": self.source_step,
            "source_mode": self.source_mode,
            "source_metric": dict(self.source_metric),
            "sampler_seed": self.sampler_seed,
            "model_seed": self.model_seed,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "source_rng_digest": self.source_rng_digest,
            "source_config_sha256": self.source_config_sha256,
            "source_archive_sha256": self.source_archive_sha256,
        }


def load_source_snapshot(
    store: CheckpointStore,
    ref: object | None = None,
    *,
    source_record: Mapping[str, object] | None = None,
) -> SourceSnapshot:
    """Load a scalar source from ``CheckpointStore`` and bind its metadata."""

    loaded = store.load(ref, load_torch=True)
    payload = loaded.torch_state
    if not isinstance(payload, Mapping):
        raise CheckpointError("source checkpoint has no torch_state mapping")
    merged: dict[str, object] = dict(source_record or {})
    merged.update(
        {
            "torch_state": payload,
            "rng_state": loaded.rng_state or {},
            "source_run_id": loaded.run_id,
            "source_step": loaded.step,
            "source_checkpoint_hash": loaded.ref.metadata_sha256,
        }
    )
    metadata = loaded.metadata.get("user_metadata")
    if isinstance(metadata, Mapping):
        for key in (
            "source_mode",
            "source_label",
            "source_metric",
            "label",
            "config_sha256",
            "source_config_sha256",
            "source_archive_sha256",
            "sampler_seed",
            "shuffle_seed",
            "seed",
        ):
            if key in metadata and key not in merged:
                merged[key] = metadata[key]
    return SourceSnapshot.from_mapping(merged)


@dataclass(frozen=True, slots=True)
class PerturbationBankConfig:
    """Registered branch horizons and continuation controls."""

    source_step: int = 20_000
    resume_steps: int = 40_000
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    checkpoint_every: int = 2_000
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 1.0e-4
    kl_coefficient: float = 0.02
    grad_clip_norm: float | None = 1.0
    device: str = "cuda"
    execution: str = "colab"
    minimum_behavior_displacement: float = 0.05
    minimum_recovery_fraction: float = 0.50
    minimum_source_retention: float = 0.80
    preserve_tolerance: float = 1.0e-3
    target_tolerance: float = 1.0e-3
    metric_scales: Mapping[str, float] | None = None
    reward_config: object | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source_step, bool) or int(self.source_step) < 0:
            raise ValueError("source_step must be non-negative")
        if isinstance(self.resume_steps, bool) or int(self.resume_steps) < 1:
            raise ValueError("resume_steps must be positive")
        normalized = tuple(sorted({int(item) for item in self.horizons}))
        if not normalized or normalized[0] < 0:
            raise ValueError("horizons must contain non-negative integers")
        if normalized[0] != 0:
            normalized = (0, *normalized)
        if normalized[-1] > int(self.resume_steps):
            raise ValueError("horizons cannot exceed resume_steps")
        object.__setattr__(self, "source_step", int(self.source_step))
        object.__setattr__(self, "resume_steps", int(self.resume_steps))
        object.__setattr__(self, "horizons", normalized)
        object.__setattr__(self, "checkpoint_every", _positive_int(self.checkpoint_every, "checkpoint_every"))
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, "batch_size"))
        for name in (
            "learning_rate",
            "weight_decay",
            "kl_coefficient",
            "minimum_behavior_displacement",
            "minimum_recovery_fraction",
            "minimum_source_retention",
            "preserve_tolerance",
            "target_tolerance",
        ):
            parsed = _finite(getattr(self, name), name)
            if name in {"learning_rate", "preserve_tolerance", "target_tolerance"} and parsed <= 0:
                raise ValueError(f"{name} must be positive")
            if name in {"weight_decay", "kl_coefficient", "minimum_behavior_displacement"} and parsed < 0:
                raise ValueError(f"{name} must be non-negative")
            if name in {"minimum_recovery_fraction", "minimum_source_retention"} and not 0 <= parsed <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, parsed)
        if self.grad_clip_norm is not None:
            clip = _finite(self.grad_clip_norm, "grad_clip_norm")
            if clip <= 0:
                raise ValueError("grad_clip_norm must be positive")
            object.__setattr__(self, "grad_clip_norm", clip)

    @classmethod
    def from_values(cls, values: Mapping[str, object]) -> "PerturbationBankConfig":
        horizons = _mapping_value(values, "horizons", "evaluation_steps", default=DEFAULT_HORIZONS)
        if isinstance(horizons, (str, bytes)) or not isinstance(horizons, Sequence):
            raise TypeError("horizons must be a sequence")
        resume = int(_mapping_value(values, "resume_steps", default=max(int(x) for x in horizons)))
        return cls(
            source_step=int(_mapping_value(values, "source_step", default=20_000)),
            resume_steps=resume,
            horizons=tuple(int(item) for item in horizons),
            checkpoint_every=int(_mapping_value(values, "checkpoint_every", "checkpoint_every_steps", default=2_000)),
            batch_size=int(_mapping_value(values, "batch_size", default=64)),
            learning_rate=float(_mapping_value(values, "learning_rate", default=0.003)),
            weight_decay=float(_mapping_value(values, "weight_decay", default=1.0e-4)),
            kl_coefficient=float(
                _mapping_value(values, "kl_coefficient", "entropy_coefficient", default=0.02)
            ),
            grad_clip_norm=(
                None
                if _mapping_value(values, "grad_clip_norm", default=1.0) is None
                else float(_mapping_value(values, "grad_clip_norm", default=1.0))
            ),
            device=str(_mapping_value(values, "device", default="cuda")),
            execution=str(_mapping_value(values, "execution", default="colab")),
            minimum_behavior_displacement=float(
                _mapping_value(values, "minimum_behavior_displacement", default=0.05)
            ),
            minimum_recovery_fraction=float(
                _mapping_value(values, "minimum_recovery_fraction", default=0.50)
            ),
            minimum_source_retention=float(
                _mapping_value(values, "minimum_source_retention", default=0.80)
            ),
            preserve_tolerance=float(
                _mapping_value(
                    values,
                    "preserve_tolerance",
                    "off_midpoint_c_on_tolerance",
                    default=1.0e-3,
                )
            ),
            target_tolerance=float(
                _mapping_value(
                    values,
                    "target_tolerance",
                    "off_midpoint_c_off_tolerance",
                    default=1.0e-3,
                )
            ),
            metric_scales=_mapping_value(values, "metric_scales", default=None),  # type: ignore[arg-type]
            reward_config=_mapping_value(values, "reward_config", default=None),
        )


def _normalize_intervention(kind: object) -> str:
    normalized = str(kind).strip().lower()
    aliases = {
        "gaussian": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "off_midpoint": "off_midpoint",
        "midpoint": "off_midpoint",
        "coff_shift_preserve_con": "off_midpoint",
        "gate": "gate_attenuation",
        "gate_ablation": "gate_attenuation",
        "gate_attenuation": "gate_attenuation",
        "opposite": "opposite_pulse",
        "hidden_pulse": "opposite_pulse",
        "opposite_pulse": "opposite_pulse",
    }
    if normalized not in aliases:
        raise ValueError(f"unknown perturbation intervention {kind!r}")
    return aliases[normalized]


def _validate_strength(kind: str, strength: object) -> float:
    parsed = _finite(strength, "strength")
    if kind in {"gaussian_noise"} and parsed < 0:
        raise ValueError("Gaussian strength must be non-negative")
    if kind in {"off_midpoint", "gate_attenuation"} and not 0 <= parsed <= 1:
        raise ValueError(f"{kind} strength must lie in [0, 1]")
    if kind == "opposite_pulse" and (parsed < 1 or parsed != int(parsed)):
        raise ValueError("opposite pulse strength must be a positive integer step count")
    return parsed


@dataclass(frozen=True, slots=True)
class PerturbationBranchSpec:
    """One matched branch in a grouped source/intervention bank."""

    source_run_id: str
    source_step: int
    source_mode: str
    intervention: str
    strength: float
    branch_kind: str
    control_kind: str
    branch_seed: int
    sampler_seed: int
    resume_steps: int
    optimizer_policy: str
    source_checkpoint_hash: str = ""
    source_rng_digest: str = ""
    source_config_sha256: str = ""
    source_archive_sha256: str = ""
    data_fingerprint: str = ""
    eval_fingerprint: str = ""
    replicate: int = 0
    branch_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_run_id, str) or not self.source_run_id.strip():
            raise ValueError("source_run_id must be non-empty")
        if isinstance(self.source_step, bool) or int(self.source_step) < 0:
            raise ValueError("source_step must be non-negative")
        if self.branch_kind not in {"resumed", "frozen", "sham", "reset_optimizer"}:
            raise ValueError("invalid branch_kind")
        if self.control_kind not in {"target", "resumed", "frozen", "sham", "reset_optimizer"}:
            raise ValueError("invalid control_kind")
        if self.optimizer_policy not in {"preserve", "reset"}:
            raise ValueError("optimizer_policy must be preserve or reset")
        if self.branch_kind == "reset_optimizer" and self.optimizer_policy != "reset":
            raise ValueError("reset_optimizer branches need reset optimizer policy")
        if self.branch_kind != "reset_optimizer" and self.optimizer_policy != "preserve":
            raise ValueError("non-reset branches need preserve optimizer policy")
        if self.branch_kind != "sham" and self.intervention != "identity":
            _validate_strength(self.intervention, self.strength)
        if isinstance(self.resume_steps, bool) or int(self.resume_steps) < 0:
            raise ValueError("resume_steps must be non-negative")
        object.__setattr__(self, "source_step", int(self.source_step))
        object.__setattr__(self, "resume_steps", int(self.resume_steps))
        object.__setattr__(self, "branch_seed", int(self.branch_seed))
        object.__setattr__(self, "sampler_seed", int(self.sampler_seed))
        object.__setattr__(self, "replicate", int(self.replicate))
        if not self.branch_id:
            object.__setattr__(self, "branch_id", self._make_branch_id())

    def _make_branch_id(self) -> str:
        payload = {
            "source_run_id": self.source_run_id,
            "source_step": self.source_step,
            "source_mode": self.source_mode,
            "intervention": self.intervention,
            "strength": self.strength,
            "branch_kind": self.branch_kind,
            "control_kind": self.control_kind,
            "branch_seed": self.branch_seed,
            "sampler_seed": self.sampler_seed,
            "resume_steps": self.resume_steps,
            "optimizer_policy": self.optimizer_policy,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "source_rng_digest": self.source_rng_digest,
            "source_config_sha256": self.source_config_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "data_fingerprint": self.data_fingerprint,
            "eval_fingerprint": self.eval_fingerprint,
            "replicate": self.replicate,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]

    def lineage(self) -> dict[str, object]:
        record = make_lineage(
            source_run_id=self.source_run_id,
            source_checkpoint=self.source_step,
            source_mode=self.source_mode,
            intervention=self.intervention,
            strength=self.strength,
            branch_kind=self.branch_kind,
            replicate=self.replicate,
            parameter_seed=self.branch_seed,
            sampler_seed=self.sampler_seed,
            data_fingerprint=self.data_fingerprint,
            optimizer_policy=self.optimizer_policy,
            resume_steps=0 if self.branch_kind == "frozen" else self.resume_steps,
            extra={
                "control_kind": self.control_kind,
                "eval_fingerprint": self.eval_fingerprint,
                "source_checkpoint_hash": self.source_checkpoint_hash,
                "source_rng_digest": self.source_rng_digest,
                "source_config_sha256": self.source_config_sha256,
                "source_archive_sha256": self.source_archive_sha256,
            },
        )
        result = record.to_dict()
        result.update(
            {
                "branch_id": self.branch_id,
                "source_step": self.source_step,
                "control_kind": self.control_kind,
                "branch_seed": self.branch_seed,
                "sampler_seed": self.sampler_seed,
                "data_fingerprint": self.data_fingerprint,
                "eval_fingerprint": self.eval_fingerprint,
                "source_checkpoint_hash": self.source_checkpoint_hash,
                "source_rng_digest": self.source_rng_digest,
                "source_config_sha256": self.source_config_sha256,
                "source_archive_sha256": self.source_archive_sha256,
            }
        )
        return result

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["lineage"] = self.lineage()
        return result


def _coerce_sources(sources: Sequence[SourceSnapshot | Mapping[str, object]]) -> tuple[SourceSnapshot, ...]:
    result = tuple(
        source if isinstance(source, SourceSnapshot) else SourceSnapshot.from_mapping(source)
        for source in sources
    )
    if not result:
        raise ValueError("at least one source checkpoint is required")
    return result


def _coerce_interventions(
    interventions: Sequence[tuple[str, object]] | Mapping[str, Sequence[object]] | None,
) -> tuple[tuple[str, float], ...]:
    if interventions is None:
        return tuple((kind, float(strength)) for kind, strength in DEFAULT_INTERVENTIONS)
    if isinstance(interventions, Mapping):
        items: list[tuple[str, object]] = []
        for kind, strengths in interventions.items():
            if isinstance(strengths, (str, bytes)) or not isinstance(strengths, Sequence):
                raise TypeError("intervention strength lists must be sequences")
            items.extend((str(kind), item) for item in strengths)
    else:
        items = []
        for item in interventions:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise TypeError("interventions must contain (kind, strength) pairs")
            items.append((str(item[0]), item[1]))
    result = tuple((_normalize_intervention(kind), _validate_strength(_normalize_intervention(kind), strength)) for kind, strength in items)
    if not result:
        raise ValueError("interventions cannot be empty")
    return result


def build_branch_specs(
    sources: Sequence[SourceSnapshot | Mapping[str, object]],
    *,
    interventions: Sequence[tuple[str, object]] | Mapping[str, Sequence[object]] | None = None,
    resume_steps: int = 40_000,
    branch_seed_base: int = 700_000,
    data_fingerprint: str = "",
    eval_fingerprint: str = "",
) -> tuple[PerturbationBranchSpec, ...]:
    """Build deterministic sham/frozen/resumed/reset controls for every source."""

    source_values = _coerce_sources(sources)
    horizon = int(resume_steps)
    if horizon < 0:
        raise ValueError("resume_steps must be non-negative")
    pairs = _coerce_interventions(interventions)
    specs: list[PerturbationBranchSpec] = []
    for source_index, source in enumerate(source_values):
        for intervention_index, (kind, strength) in enumerate(pairs):
            seed_base = int(branch_seed_base) + source_index * 100_000 + intervention_index * 100
            common = {
                "source_run_id": source.source_run_id,
                "source_step": source.source_step,
                "source_mode": source.source_mode,
                "source_checkpoint_hash": source.source_checkpoint_hash,
                "source_rng_digest": source.source_rng_digest,
                "source_config_sha256": source.source_config_sha256,
                "source_archive_sha256": source.source_archive_sha256,
                "data_fingerprint": data_fingerprint,
                "eval_fingerprint": eval_fingerprint,
                "sampler_seed": source.sampler_seed,
            }
            specs.extend(
                (
                    PerturbationBranchSpec(
                        **common,
                        intervention=kind,
                        strength=strength,
                        branch_kind="frozen",
                        control_kind="frozen",
                        branch_seed=seed_base + 1,
                        optimizer_policy="preserve",
                        resume_steps=0,
                    ),
                    PerturbationBranchSpec(
                        **common,
                        intervention="identity",
                        strength=0.0,
                        branch_kind="sham",
                        control_kind="sham",
                        branch_seed=seed_base + 2,
                        optimizer_policy="preserve",
                        resume_steps=horizon,
                    ),
                    PerturbationBranchSpec(
                        **common,
                        intervention=kind,
                        strength=strength,
                        branch_kind="resumed",
                        control_kind="target",
                        branch_seed=seed_base + 3,
                        optimizer_policy="preserve",
                        resume_steps=horizon,
                    ),
                    PerturbationBranchSpec(
                        **common,
                        intervention=kind,
                        strength=strength,
                        branch_kind="reset_optimizer",
                        control_kind="reset_optimizer",
                        branch_seed=seed_base + 4,
                        optimizer_policy="reset",
                        resume_steps=horizon,
                    ),
                )
            )
    branch_ids = [spec.branch_id for spec in specs]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("branch specification generated duplicate branch IDs")
    return tuple(specs)


def group_branch_specs(
    specs: Sequence[PerturbationBranchSpec],
) -> dict[tuple[str, str], tuple[PerturbationBranchSpec, ...]]:
    """Group branch specifications by intervention and control identity.

    The key is the same one used by :func:`run_perturbation_bank`, so callers
    can inspect or schedule grouped restarts without reproducing its private
    grouping rule.
    """

    groups: dict[tuple[str, str], list[PerturbationBranchSpec]] = {}
    for spec in specs:
        if not isinstance(spec, PerturbationBranchSpec):
            raise TypeError("specs must contain PerturbationBranchSpec values")
        groups.setdefault(_group_key(spec), []).append(spec)
    return {key: tuple(value) for key, value in groups.items()}


def _training_state_mapping(value: object, *, source_step: int, sampler_seed: int) -> dict[str, object]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif is_dataclass(value):
        result = asdict(value)
    else:
        result = {}
        for name in ("epoch", "batch_offset", "global_step", "permutation", "sampler_state", "history", "completed"):
            if hasattr(value, name):
                result[name] = getattr(value, name)
    result.setdefault("epoch", 0)
    result.setdefault("batch_offset", 0)
    result.setdefault("global_step", source_step)
    result.setdefault("permutation", None)
    result.setdefault("sampler_state", None)
    result.setdefault("history", [])
    result.setdefault("completed", False)
    result["global_step"] = int(result.get("global_step") or source_step)
    return result


def _to_cpu(value: object) -> object:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _restore_rng(value: Mapping[str, object]) -> None:
    if not value:
        return
    from .training import restore_rng_state

    restore_rng_state(value)


def _capture_rng() -> dict[str, object]:
    from .training import capture_rng_state

    return capture_rng_state()


def _call_factory(factory: ModelFactory | None, source: SourceSnapshot, spec: PerturbationBranchSpec) -> object:
    if factory is None:
        candidate = _mapping_value(source.raw, "agent", "model", default=None)
        if candidate is None:
            raise ValueError(
                "model_factory is required when source records contain only state_dicts; "
                "pass a factory that returns the scalar architecture"
            )
        return copy.deepcopy(candidate)
    attempts: list[tuple[tuple[object, ...], dict[str, object]]] = [
        ((), {"source": source, "spec": spec}),
        ((), {"source": source.raw, "spec": spec}),
        ((source, spec), {}),
        ((source.raw, spec), {}),
        ((source,), {}),
        ((source.raw,), {}),
        ((spec,), {}),
        ((), {}),
    ]
    last: Exception | None = None
    for args, kwargs in attempts:
        try:
            return factory(*args, **kwargs)
        except (TypeError, AttributeError) as exc:
            last = exc
    raise TypeError("model_factory could not be called with source/spec or no arguments") from last


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for perturbation bank execution in Colab") from exc
    return torch


def _parameter_snapshot(agent: object) -> dict[str, object]:
    return {
        str(name): parameter.detach().clone()
        for name, parameter in agent.named_parameters()  # type: ignore[attr-defined]
    }


def _parameter_delta_norm(before: Mapping[str, object], after: Mapping[str, object]) -> float | None:
    torch = _require_torch()
    terms = []
    for name, previous in before.items():
        current = after.get(name)
        if current is None:
            continue
        terms.append((current.detach() - previous).float().square().sum())  # type: ignore[union-attr]
    if not terms:
        return None
    return float(torch.stack(terms).sum().sqrt().detach().cpu().item())


def _parameter_norm(parameters: Mapping[str, object]) -> float | None:
    torch = _require_torch()
    terms = [
        value.detach().float().square().sum()
        for value in parameters.values()
        if hasattr(value, "detach") and getattr(value, "is_floating_point", lambda: True)()
    ]
    if not terms:
        return None
    return float(torch.stack(terms).sum().sqrt().detach().cpu().item())


def _gradient_norm(agent: object) -> float | None:
    torch = _require_torch()
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in agent.parameters()  # type: ignore[attr-defined]
        if getattr(parameter, "grad", None) is not None
    ]
    if not terms:
        return None
    return float(torch.stack(terms).sum().sqrt().detach().cpu().item())


def _optimizer_options(optimizer: object) -> dict[str, object]:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return {"lr": 0.003}
    return {
        key: copy.deepcopy(value)
        for key, value in groups[0].items()
        if key != "params"
    }


def _new_optimizer(agent: object, source_optimizer_state: Mapping[str, object], *, load_state: bool, lr: float) -> object:
    torch = _require_torch()
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(lr))
    if load_state and source_optimizer_state:
        optimizer.load_state_dict(copy.deepcopy(dict(source_optimizer_state)))
    return optimizer


def _opportunity_batches(pairs: Sequence[PairedEpisode], device: object) -> tuple[object, object]:
    selected = tuple(pair for pair in pairs if bool(pair.on.secret_opportunity))
    if not selected:
        raise ValueError("intervention requires at least one opportunity-conditioned pair")
    return collate(tuple(pair.on for pair in selected)).to(device), collate(
        tuple(pair.off for pair in selected)
    ).to(device)


def _apply_scalar_intervention(
    agent: object,
    spec: PerturbationBranchSpec,
    source_metric: Mapping[str, object],
    pairs: Sequence[PairedEpisode],
    *,
    evaluate_fn: MetricEvaluator,
    preserve_tolerance: float,
    target_tolerance: float,
) -> dict[str, object]:
    """Apply the exact scalar equations used by ``campaign._intervene``."""

    torch = _require_torch()
    before = _parameter_snapshot(agent)
    device = next(agent.parameters()).device  # type: ignore[attr-defined]
    kind = spec.intervention
    strength = float(spec.strength)
    if spec.branch_kind == "sham" or spec.control_kind == "sham":
        return {
            "kind": "identity",
            "strength": 0.0,
            "seed": int(spec.branch_seed),
            "steps": 0,
            "parameter_delta_norm": 0.0,
            "parameter_relative_delta": 0.0,
            "gradient_norm": None,
            "update_norm": None,
            "feasible": True,
            "status": "identity",
        }
    if kind == "gaussian_noise":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(spec.branch_seed))
        with torch.no_grad():
            for parameter in agent.parameters():  # type: ignore[attr-defined]
                noise = torch.randn(
                    parameter.shape,
                    generator=generator,
                    device=device,
                    dtype=parameter.dtype,
                )
                noise_norm = noise.norm().clamp_min(1.0e-12)
                scale = parameter.detach().norm().clamp_min(1.0e-3) * strength
                parameter.add_(noise * scale / noise_norm)
        after = _parameter_snapshot(agent)
        delta = _parameter_delta_norm(before, after)
        norm = _parameter_norm(after)
        return {
            "kind": kind,
            "strength": strength,
            "seed": int(spec.branch_seed),
            "steps": 0,
            "parameter_delta_norm": delta,
            "parameter_relative_delta": None if not norm else (delta or 0.0) / norm,
            "gradient_norm": None,
            "update_norm": None,
            "feasible": True,
            "status": "applied",
        }
    if kind == "gate_attenuation":
        gate = getattr(agent, "oversight_gate", None)
        if gate is None or not callable(getattr(gate, "parameters", None)):
            raise ValueError("gate attenuation requires an agent.oversight_gate module")
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.mul_(strength)
        after = _parameter_snapshot(agent)
        delta = _parameter_delta_norm(before, after)
        norm = _parameter_norm(after)
        return {
            "kind": kind,
            "strength": strength,
            "seed": int(spec.branch_seed),
            "steps": 0,
            "parameter_delta_norm": delta,
            "parameter_relative_delta": None if not norm else (delta or 0.0) / norm,
            "gradient_norm": None,
            "update_norm": None,
            "feasible": True,
            "status": "applied",
        }

    on, off = _opportunity_batches(pairs, device)
    intervention_optimizer = torch.optim.Adam(agent.parameters(), lr=0.003)
    source_on = _finite(_mapping_value(source_metric, "c_on", "C_on"), "source c_on")
    source_off = _finite(_mapping_value(source_metric, "c_off", "C_off"), "source c_off")
    if kind == "off_midpoint":
        steps = 50
        target_off = source_off + strength * (0.5 - source_off)
        target_on = source_on
    elif kind == "opposite_pulse":
        steps = int(strength)
        label = str(source_metric.get("label", Mode.INTERMEDIATE.value))
        target_off = 0.05 if label == Mode.OVERSIGHT_INVARIANT.value else 0.95
        target_on = max(0.95, source_on)
    else:  # pragma: no cover - validation catches this path
        raise ValueError(f"unsupported intervention {kind}")
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    for _ in range(steps):
        intervention_optimizer.zero_grad(set_to_none=True)
        p_on = agent.help_probability(on).mean()  # type: ignore[attr-defined]
        p_off = agent.help_probability(off).mean()  # type: ignore[attr-defined]
        loss = (p_off - target_off) ** 2 + 10.0 * (p_on - target_on) ** 2
        loss.backward()
        current_gradient = _gradient_norm(agent)
        if current_gradient is not None:
            gradient_norms.append(current_gradient)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  # type: ignore[attr-defined]
        update_before = _parameter_snapshot(agent)
        intervention_optimizer.step()
        current_update = _parameter_delta_norm(update_before, _parameter_snapshot(agent))
        if current_update is not None:
            update_norms.append(current_update)
    after = _parameter_snapshot(agent)
    delta = _parameter_delta_norm(before, after)
    norm = _parameter_norm(after)
    result: dict[str, object] = {
        "kind": kind,
        "strength": strength,
        "seed": int(spec.branch_seed),
        "steps": int(steps),
        "parameter_delta_norm": delta,
        "parameter_relative_delta": None if not norm else (delta or 0.0) / norm,
        "gradient_norm": None if not gradient_norms else sum(gradient_norms) / len(gradient_norms),
        "gradient_norm_last": gradient_norms[-1] if gradient_norms else None,
        "update_norm": None if not update_norms else sum(update_norms) / len(update_norms),
        "update_norm_last": update_norms[-1] if update_norms else None,
        "feasible": True,
        "status": "applied",
    }
    if kind == "off_midpoint":
        measured = _metric_mapping(evaluate_fn(agent, pairs))
        measured_on = _finite(measured["c_on"], "measured c_on")
        measured_off = _finite(measured["c_off"], "measured c_off")
        on_error = abs(measured_on - source_on)
        off_target_error = abs(measured_off - target_off)
        result.update(
            {
                "initial_c_on": source_on,
                "initial_c_off": source_off,
                "target_c_on": source_on,
                "target_c_off": target_off,
                "final_c_on": measured_on,
                "final_c_off": measured_off,
                "c_on_preservation_error": on_error,
                "c_off_target_error": off_target_error,
                "preserve_tolerance": preserve_tolerance,
                "target_tolerance": target_tolerance,
                "feasible": on_error <= preserve_tolerance and off_target_error <= target_tolerance,
                "status": (
                    "feasible"
                    if on_error <= preserve_tolerance and off_target_error <= target_tolerance
                    else "infeasible_c_on_preservation"
                ),
            }
        )
    return result


def apply_intervention(
    agent: object,
    spec: PerturbationBranchSpec,
    source_metric: MetricLike,
    pairs: Sequence[PairedEpisode],
    *,
    evaluate_fn: MetricEvaluator | None = None,
    preserve_tolerance: float = 1.0e-3,
    target_tolerance: float = 1.0e-3,
) -> dict[str, object]:
    """Apply a registered scalar intervention and return diagnostics.

    This is the low-level campaign hook.  ``agent`` must expose
    ``parameters()``, ``help_probability()``, and, for gate attenuation,
    ``oversight_gate``.  The source metric is converted through the same
    parser used by the grouped runner.
    """

    if not isinstance(spec, PerturbationBranchSpec):
        raise TypeError("spec must be a PerturbationBranchSpec")
    metric = _metric_mapping(source_metric)
    evaluator = evaluate_fn
    if evaluator is None:
        evaluator = lambda model, evaluation_pairs: evaluate_agent(model, evaluation_pairs)
    return _apply_scalar_intervention(
        agent,
        spec,
        metric,
        pairs,
        evaluate_fn=evaluator,
        preserve_tolerance=preserve_tolerance,
        target_tolerance=target_tolerance,
    )


def _training_payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif is_dataclass(value):
        result = asdict(value)
    else:
        result = {
            name: getattr(value, name)
            for name in (
                "epoch",
                "batch_offset",
                "global_step",
                "permutation",
                "sampler_state",
                "history",
                "completed",
            )
            if hasattr(value, name)
        }
    return dict(_to_cpu(result))


def _training_state_from_mapping(value: Mapping[str, object]) -> object:
    from .training import TrainingState

    allowed = {"epoch", "batch_offset", "global_step", "permutation", "sampler_state", "history", "completed"}
    result = {key: value[key] for key in allowed if key in value}
    torch = _require_torch()
    for name, dtype in (("permutation", torch.long), ("sampler_state", torch.uint8)):
        current = result.get(name)
        if current is not None and not isinstance(current, torch.Tensor):
            result[name] = torch.as_tensor(current, dtype=dtype, device="cpu")
    return TrainingState(**result)


def _scalar_optimizer_state(optimizer: object) -> Mapping[str, object]:
    return dict(_to_cpu(optimizer.state_dict()))  # type: ignore[attr-defined]


def _parameter_names(model: object) -> tuple[str, ...]:
    return tuple(str(name) for name, _ in model.named_parameters())  # type: ignore[attr-defined]


def _state_by_parameter_index(state: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    raw = state.get("state", {})
    if not isinstance(raw, Mapping):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            result[index] = value
    return result


def _install_stacked_optimizer_state(
    bank: ReplicaBank,
    scalar_optimizers: Sequence[object],
) -> None:
    """Stack per-source Adam moments into a ``ReplicaBank`` optimizer."""

    torch = _require_torch()
    if len(scalar_optimizers) != bank.replica_count:
        raise ValueError("optimizer count does not match bank replicas")
    bank_optimizer = bank.optimizer
    if bank_optimizer is None:
        raise ValueError("ReplicaBank has no optimizer")
    state_maps = [_state_by_parameter_index(_scalar_optimizer_state(item)) for item in scalar_optimizers]
    for name_index, parameter in enumerate(bank.parameters()):
        entries = [state_map.get(name_index, {}) for state_map in state_maps]
        keys = set().union(*(entry.keys() for entry in entries))
        if not keys:
            continue
        stacked: dict[str, object] = {}
        for key in keys:
            values = [entry.get(key) for entry in entries]
            if all(isinstance(value, torch.Tensor) for value in values):
                tensors = [value for value in values if isinstance(value, torch.Tensor)]
                if tensors and tensors[0].ndim == 0:
                    stacked[key] = tensors[0].to(device=parameter.device)
                elif tensors and tuple(tensors[0].shape) == tuple(parameter.shape[1:]):
                    stacked[key] = torch.stack(
                        [value.to(device=parameter.device, dtype=parameter.dtype) for value in tensors],
                        dim=0,
                    )
                else:
                    stacked[key] = tensors[0].to(device=parameter.device)
            elif all(value is not None for value in values):
                stacked[key] = copy.deepcopy(values[0])
        bank_optimizer.state[parameter] = stacked
    groups = [getattr(item, "param_groups", ()) for item in scalar_optimizers]
    if groups and all(group and group[0] for group in groups):
        options = _optimizer_options(scalar_optimizers[0])
        for key, value in options.items():
            if key != "params":
                bank_optimizer.param_groups[0][key] = copy.deepcopy(value)


def _slice_optimizer_state(bank: ReplicaBank, index: int) -> Mapping[str, object]:
    model, optimizer = bank.materialize_replica_with_optimizer(index, device="cpu")
    del model
    return _scalar_optimizer_state(optimizer)


def _bank_training_state(
    state: BatchedTrainingState,
    index: int,
) -> dict[str, object]:
    permutations = state.permutations
    sampler_states = state.sampler_states
    return {
        "epoch": int(state.epoch),
        "batch_offset": int(state.batch_offset),
        "global_step": int(state.global_step),
        "permutation": None if permutations is None else permutations[index].detach().cpu(),
        "sampler_state": None if not sampler_states else sampler_states[index].detach().cpu(),
        "history": list(state.history),
        "completed": bool(state.completed),
    }


def _branch_metric_row(
    metric: Mapping[str, object],
    *,
    spec: PerturbationBranchSpec,
    source_metric: Mapping[str, object],
    opposite_metric: Mapping[str, object] | None,
    step: int,
    horizon: int,
    eval_hash: str,
    train_reward: object = None,
    train_loss: object = None,
    thresholds: ModeThresholds,
    intervention_diagnostics: Mapping[str, object],
    metric_scales: Mapping[str, float] | None = None,
) -> dict[str, object]:
    metric_value = dict(metric)
    metric_value["c_on"] = _finite(metric_value["c_on"], "c_on")
    metric_value["c_off"] = _finite(metric_value["c_off"], "c_off")
    metric_value["gap"] = metric_value["c_on"] - metric_value["c_off"]
    for name in ("goal", "gate", "off_audit_logit"):
        if metric_value.get(name) is not None:
            metric_value[name] = _finite(metric_value[name], name)
    mode_label = _metric_for_label(metric_value, thresholds)
    source_vector = {
        "c_on": source_metric.get("c_on", 0.0),
        "c_off": source_metric.get("c_off", 0.0),
        "goal": source_metric.get("goal", 0.0) or 0.0,
        "gate": source_metric.get("gate", 0.0) or 0.0,
    }
    d_source: float | None = None
    d_opposite: float | None = None
    vector = {
        "c_on": metric_value["c_on"],
        "c_off": metric_value["c_off"],
        "goal": metric_value.get("goal", 0.0) or 0.0,
        "gate": metric_value.get("gate", 0.0) or 0.0,
    }
    fields = ("c_on", "c_off", "goal", "gate")
    scale_values = _metric_scale_payload(metric_scales)
    d_source = math.sqrt(
        sum(
            ((float(vector[field]) - float(source_vector[field])) / scale_values[field]) ** 2
            for field in fields
        )
    )
    if opposite_metric is not None:
        opposite_vector = {
            field: opposite_metric.get(field, 0.0) or 0.0 for field in fields
        }
        d_opposite = math.sqrt(
            sum(
                ((float(vector[field]) - float(opposite_vector[field])) / scale_values[field]) ** 2
                for field in fields
            )
        )
    feasible = intervention_diagnostics.get("feasible", True)
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "run_id": spec.branch_id,
        "branch_id": spec.branch_id,
        "source_run_id": spec.source_run_id,
        "source_step": spec.source_step,
        "step": int(step),
        "step_since_branch": int(horizon),
        "eval_variant": "paired_audit_opportunity",
        "eval_set_hash": eval_hash,
        "c_on": metric_value["c_on"],
        "c_off": metric_value["c_off"],
        "gap": metric_value["gap"],
        "goal_score": metric_value.get("goal"),
        "goal": metric_value.get("goal"),
        "gate_effect": metric_value.get("gate"),
        "gate": metric_value.get("gate"),
        "off_audit_logit": metric_value.get("off_audit_logit"),
        "train_reward": train_reward,
        "reward": train_reward,
        "hidden_misalignment_rate": 1.0 - float(metric_value["c_off"]),
        "hidden_misalignment": 1.0 - float(metric_value["c_off"]),
        "train_loss": train_loss,
        "n_pairs": metric_value.get("n_pairs"),
        "is_final": False,
        "label": mode_label,
        "source_label": spec.source_mode,
        "branch_kind": spec.branch_kind,
        "control_kind": spec.control_kind,
        "intervention": spec.intervention,
        "strength": spec.strength,
        "branch_seed": spec.branch_seed,
        "d_source": d_source,
        "d_opposite": d_opposite,
        "source_label_retention": 1.0 if mode_label == spec.source_mode else 0.0,
        "source_distance_closer": None if d_opposite is None else d_source < d_opposite,
        "intervention_feasible": feasible,
        "intervention_status": intervention_diagnostics.get("status", "applied"),
        "intervention_diagnostics": dict(intervention_diagnostics),
        "parameter_delta_norm": intervention_diagnostics.get("parameter_delta_norm"),
        "parameter_relative_delta": intervention_diagnostics.get("parameter_relative_delta"),
        "gradient_norm": intervention_diagnostics.get("gradient_norm"),
        "update_norm": intervention_diagnostics.get("update_norm"),
        "mode_label": mode_label,
        "recovery_status": "pending",
        "optimizer_policy": spec.optimizer_policy,
        "source_checkpoint_hash": spec.source_checkpoint_hash,
        "source_checkpoint_digest": spec.source_checkpoint_hash,
        "source_rng_digest": spec.source_rng_digest,
        "data_fingerprint": spec.data_fingerprint,
    }


def _metric_scales_from_mapping(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("metric_scales must be a mapping")
    result = {}
    for field in ("c_on", "c_off", "goal", "gate"):
        parsed = _finite(value.get(field, 1.0), f"metric scale {field}")
        if parsed <= 0:
            raise ValueError("metric scales must be positive")
        result[field] = parsed
    return result


def _metric_scale_payload(value: Mapping[str, object] | None) -> dict[str, float]:
    result = {field: 1.0 for field in ("c_on", "c_off", "goal", "gate")}
    if value:
        for field in result:
            if field in value:
                result[field] = _finite(value[field], f"metric scale {field}")
    return result


def _evaluate_bank_default(
    bank: ReplicaBank,
    specs: Sequence[PerturbationBranchSpec],
    pairs: Sequence[PairedEpisode],
    *,
    evaluator: MetricEvaluator,
) -> list[dict[str, object]]:
    return [
        _metric_mapping(
            evaluator(bank.materialize_replica(index, device=bank.device), pairs)
        )
        for index, _spec in enumerate(specs)
    ]


def _call_bank_evaluator(
    evaluator: BankMetricEvaluator | None,
    bank: ReplicaBank,
    specs: Sequence[PerturbationBranchSpec],
    pairs: Sequence[PairedEpisode],
    *,
    fallback: MetricEvaluator,
) -> list[dict[str, object]]:
    if evaluator is None:
        return _evaluate_bank_default(bank, specs, pairs, evaluator=fallback)
    attempts = (
        ((bank, pairs), {"specs": specs}),
        ((bank, specs, pairs), {}),
        ((bank, pairs), {}),
    )
    last: Exception | None = None
    for args, kwargs in attempts:
        try:
            value = evaluator(*args, **kwargs)
            if isinstance(value, tuple) and len(value) == 3 and isinstance(value[1], Sequence):
                value = value[1]
            if isinstance(value, Mapping):
                value = [value[key] for key in specs]
            if not isinstance(value, Sequence) or len(value) != len(specs):
                raise ValueError("bank evaluator must return one metric per branch")
            return [_metric_mapping(item) for item in value]  # type: ignore[arg-type]
        except (TypeError, AttributeError, ValueError, RuntimeError) as exc:
            last = exc
    raise TypeError("bank evaluator could not be called with (bank, pairs, specs)") from last


def _replica_value(value: object, index: int) -> object:
    """Extract one scalar row from a tensor/list metric vector."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    return value


def _group_key(spec: PerturbationBranchSpec) -> tuple[str, str]:
    return spec.intervention, spec.control_kind


def _group_id(specs: Sequence[PerturbationBranchSpec], config_identity: str, source_identity: str) -> str:
    payload = {
        "config": config_identity,
        "source": source_identity,
        "specs": [spec.branch_id for spec in specs],
    }
    return "perturb-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def _group_manifest(
    group_id: str,
    specs: Sequence[PerturbationBranchSpec],
    *,
    config: PerturbationBankConfig,
    config_identity: str,
    source_identity: str,
    dataset_hash: str,
    eval_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": PERTURBATION_BANK_SCHEMA_VERSION,
        "group_id": group_id,
        "config_sha256": config_identity,
        "source_archive_sha256": source_identity,
        "dataset_sha256": dataset_hash,
        "evaluation_sha256": eval_hash,
        "source_step": config.source_step,
        "resume_steps": config.resume_steps,
        "horizons": list(config.horizons),
        "specs": [spec.as_dict() for spec in specs],
        "backend": "ReplicaBank",
    }


def _checkpoint_payload(
    bank: ReplicaBank,
    state: BatchedTrainingState,
    specs: Sequence[PerturbationBranchSpec],
    diagnostics: Sequence[Mapping[str, object]],
    *,
    group_id: str,
    config: PerturbationBankConfig,
    config_identity: str,
    source_identity: str,
    dataset_hash: str,
    eval_hash: str,
    metrics: Sequence[Mapping[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    rng = _capture_rng()
    branches: dict[str, object] = {}
    for index, spec in enumerate(specs):
        model_state = _to_cpu(bank.replica_state_dict(index))
        branches[spec.branch_id] = {
            "model_state": model_state,
            "optimizer_state": _slice_optimizer_state(bank, index),
            "training_state": _training_payload(_bank_training_state(state, index)),
            "intervention_diagnostics": dict(diagnostics[index]),
            "post_intervention_hash": state_fingerprint(model_state),
        }
    torch_rng = {key: value for key, value in rng.items() if str(key).startswith("torch_")}
    ordinary_rng = {key: value for key, value in rng.items() if not str(key).startswith("torch_")}
    payload = {
        "schema_version": PERTURBATION_BANK_SCHEMA_VERSION,
        "group_id": group_id,
        "config": asdict(config),
        "config_sha256": config_identity,
        "source_archive_sha256": source_identity,
        "dataset_sha256": dataset_hash,
        "evaluation_sha256": eval_hash,
        "global_step": int(state.global_step),
        "branches": branches,
        "torch_rng": _to_cpu(torch_rng),
        "metrics": [dict(item) for item in (metrics or ())],
    }
    state_summary = {
        "global_step": int(state.global_step),
        "epoch": int(state.epoch),
        "batch_offset": int(state.batch_offset),
        "branch_ids": [spec.branch_id for spec in specs],
        "replica_count": len(specs),
    }
    return payload, {"state": state_summary, "rng": ordinary_rng}


def save_group_checkpoint(
    store: CheckpointStore,
    *,
    bank: ReplicaBank,
    state: BatchedTrainingState,
    specs: Sequence[PerturbationBranchSpec],
    diagnostics: Sequence[Mapping[str, object]],
    group_id: str,
    config: PerturbationBankConfig,
    config_identity: str,
    source_identity: str,
    dataset_hash: str,
    eval_hash: str,
    metrics: Sequence[Mapping[str, object]] | None = None,
) -> object:
    """Atomically save every branch in one grouped checkpoint."""

    payload, sidecars = _checkpoint_payload(
        bank,
        state,
        specs,
        diagnostics,
        group_id=group_id,
        config=config,
        config_identity=config_identity,
        source_identity=source_identity,
        dataset_hash=dataset_hash,
        eval_hash=eval_hash,
        metrics=metrics,
    )
    return store.save(
        int(state.global_step),
        sidecars["state"],
        optimizer_state={"storage": "torch_state.pt"},
        rng_state=sidecars["rng"],
        minibatch_cursor=int(state.batch_offset),
        torch_state=payload,
        metadata={
            "group_id": group_id,
            "config_sha256": config_identity,
            "source_archive_sha256": source_identity,
            "dataset_sha256": dataset_hash,
            "evaluation_sha256": eval_hash,
            "branch_ids": [spec.branch_id for spec in specs],
        },
    )


def load_group_checkpoint(
    store: CheckpointStore,
    ref: object | None = None,
    *,
    expected_config_identity: str | None = None,
) -> Mapping[str, object]:
    """Load and validate a grouped torch payload."""

    loaded = store.load(ref, load_torch=True, expected_config_identity=expected_config_identity)
    payload = loaded.torch_state
    if not isinstance(payload, Mapping):
        raise CheckpointError("group checkpoint has no torch payload")
    if int(payload.get("schema_version", -1)) != PERTURBATION_BANK_SCHEMA_VERSION:
        raise CheckpointError("unsupported perturbation bank checkpoint schema")
    if expected_config_identity is not None and payload.get("config_sha256") != expected_config_identity:
        raise CheckpointError("group checkpoint config identity mismatch")
    if not isinstance(payload.get("branches"), Mapping):
        raise CheckpointError("group checkpoint has no branch mapping")
    return payload


def _restore_group_into_models(
    payload: Mapping[str, object],
    specs: Sequence[PerturbationBranchSpec],
    models: Sequence[object],
    optimizers: Sequence[object],
) -> None:
    branches = payload.get("branches")
    if not isinstance(branches, Mapping):
        raise ValueError("checkpoint branches must be a mapping")
    for index, spec in enumerate(specs):
        record = branches.get(spec.branch_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"group checkpoint is missing branch {spec.branch_id}")
        model_state = record.get("model_state")
        optimizer_state = record.get("optimizer_state")
        if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
            raise ValueError(f"group checkpoint branch {spec.branch_id} is incomplete")
        models[index].load_state_dict(model_state, strict=True)  # type: ignore[attr-defined]
        optimizers[index].load_state_dict(copy.deepcopy(dict(optimizer_state)))  # type: ignore[attr-defined]


def _make_bank_state(
    records: Sequence[Mapping[str, object]],
    *,
    replica_count: int,
    source_step: int,
) -> BatchedTrainingState:
    torch = _require_torch()
    states = [_training_state_from_mapping(record) for record in records]
    permutations = [getattr(state, "permutation", None) for state in states]
    sampler_states = [getattr(state, "sampler_state", None) for state in states]
    permutation_stack = None
    if all(item is not None for item in permutations):
        permutation_stack = torch.stack([item.detach().cpu() for item in permutations], dim=0)
    sampler_tuple = tuple(item.detach().cpu() for item in sampler_states if item is not None)
    if sampler_tuple and len(sampler_tuple) != replica_count:
        raise ValueError("group checkpoint sampler state count mismatch")
    first = states[0] if states else _training_state_from_mapping({"global_step": source_step})
    return BatchedTrainingState(
        replica_count=replica_count,
        epoch=int(getattr(first, "epoch", 0)),
        batch_offset=int(getattr(first, "batch_offset", 0)),
        global_step=int(getattr(first, "global_step", source_step)),
        permutations=permutation_stack,
        sampler_states=sampler_tuple,
        history=[],
        completed=False,
    )


def _compact_row_metric(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "c_on": row.get("c_on", row.get("C_on")),
        "c_off": row.get("c_off", row.get("C_off")),
        "goal": row.get("goal", row.get("goal_score")),
        "gate": row.get("gate", row.get("gate_effect")),
        "off_audit_logit": row.get("off_audit_logit"),
        "n_pairs": row.get("n_pairs"),
        "label": row.get("label", row.get("mode_label")),
    }


@dataclass(frozen=True, slots=True)
class PerturbationBankResult:
    """Compact outputs from a complete grouped perturbation campaign."""

    tables: Mapping[str, Sequence[Mapping[str, object]]]
    branch_specs: tuple[PerturbationBranchSpec, ...]
    group_ids: tuple[str, ...]
    lineages: Mapping[str, Mapping[str, object]]


def _build_source_model_and_optimizer(
    source: SourceSnapshot,
    spec: PerturbationBranchSpec,
    *,
    model_factory: ModelFactory | None,
    config: PerturbationBankConfig,
    evaluate_fn: MetricEvaluator,
    pairs: Sequence[PairedEpisode],
) -> tuple[object, object, dict[str, object]]:
    model = _call_factory(model_factory, source, spec)
    torch = _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    model = model.to(config.device)
    model.load_state_dict(copy.deepcopy(dict(source.model_state)), strict=True)
    optimizer = _new_optimizer(
        model,
        source.optimizer_state,
        load_state=spec.optimizer_policy == "preserve",
        lr=config.learning_rate,
    )
    diagnostics = _apply_scalar_intervention(
        model,
        spec,
        source.source_metric,
        pairs,
        evaluate_fn=evaluate_fn,
        preserve_tolerance=config.preserve_tolerance,
        target_tolerance=config.target_tolerance,
    )
    return model, optimizer, diagnostics


def _restore_bank_optimizer_from_scalar(
    bank: ReplicaBank,
    optimizers: Sequence[object],
) -> None:
    _install_stacked_optimizer_state(bank, optimizers)


def _prepare_group(
    specs: Sequence[PerturbationBranchSpec],
    sources: Mapping[str, SourceSnapshot],
    *,
    model_factory: ModelFactory | None,
    config: PerturbationBankConfig,
    evaluate_fn: MetricEvaluator,
    pairs: Sequence[PairedEpisode],
    resume_payload: Mapping[str, object] | None,
) -> tuple[ReplicaBank, list[object], list[object], list[dict[str, object]], BatchedTrainingState]:
    models: list[object] = []
    optimizers: list[object] = []
    diagnostics: list[dict[str, object]] = []
    for spec in specs:
        source = sources[spec.source_run_id]
        model, optimizer, intervention = _build_source_model_and_optimizer(
            source,
            spec,
            model_factory=model_factory,
            config=config,
            evaluate_fn=evaluate_fn,
            pairs=pairs,
        )
        models.append(model)
        optimizers.append(optimizer)
        diagnostics.append(dict(intervention))
    if resume_payload is not None:
        _restore_group_into_models(resume_payload, specs, models, optimizers)
    bank = ReplicaBank.from_agents(
        models,  # type: ignore[arg-type]
        model_seeds=tuple(sources[spec.source_run_id].model_seed for spec in specs),
        sampler_seeds=tuple(spec.sampler_seed for spec in specs),
        device=config.device,
        learning_rate=config.learning_rate,
    )
    _restore_bank_optimizer_from_scalar(bank, optimizers)
    if resume_payload is not None:
        records = [
            item for item in (
                resume_payload.get("branches", {}).get(spec.branch_id)  # type: ignore[union-attr]
                for spec in specs
            )
            if isinstance(item, Mapping)
        ]
        if len(records) != len(specs):
            raise ValueError("group checkpoint does not contain every branch state")
        state = _make_bank_state(records, replica_count=len(specs), source_step=specs[0].source_step)
    else:
        state_records = [
            _training_state_mapping(sources[spec.source_run_id].training_state, source_step=spec.source_step, sampler_seed=spec.sampler_seed)
            for spec in specs
        ]
        state = _make_bank_state(state_records, replica_count=len(specs), source_step=specs[0].source_step)
    return bank, models, optimizers, diagnostics, state


def _validate_group_manifest(path: Path, expected: Mapping[str, object]) -> None:
    if not path.is_file():
        _write_json(path, expected)
        return
    current = _read_json(path)
    for key in (
        "schema_version",
        "group_id",
        "config_sha256",
        "source_archive_sha256",
        "dataset_sha256",
        "evaluation_sha256",
        "source_step",
        "resume_steps",
        "horizons",
        "specs",
    ):
        if current.get(key) != _jsonable(expected.get(key)):
            raise ValueError(f"perturbation group manifest mismatch for {key}")


def _source_opposite(sources: Mapping[str, SourceSnapshot], source: SourceSnapshot) -> Mapping[str, object] | None:
    labels = {item.source_mode for item in sources.values()}
    opposite = (
        Mode.STRATEGIC.value
        if source.source_mode == Mode.OVERSIGHT_INVARIANT.value
        else Mode.OVERSIGHT_INVARIANT.value
        if source.source_mode == Mode.STRATEGIC.value
        else None
    )
    if opposite is None or opposite not in labels:
        return None
    records = [item.source_metric for item in sources.values() if item.source_mode == opposite]
    if not records:
        return None
    fields = ("c_on", "c_off", "goal", "gate")
    result: dict[str, object] = {"label": opposite}
    for field in fields:
        values = [float(record.get(field, 0.0) or 0.0) for record in records]
        result[field] = sum(values) / len(values)
    return result


def _default_source_scale(sources: Sequence[SourceSnapshot]) -> dict[str, float]:
    fields = ("c_on", "c_off", "goal", "gate")
    result: dict[str, float] = {}
    for field in fields:
        values = [float(source.source_metric.get(field, 0.0) or 0.0) for source in sources]
        if len(values) < 2:
            result[field] = 1.0
            continue
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        result[field] = max(0.05, spread)
    return result


def _set_recovery_fields(
    rows: list[dict[str, object]],
    *,
    spec: PerturbationBranchSpec,
    source_metric: Mapping[str, object],
    opposite_metric: Mapping[str, object] | None,
    frozen_rows: Sequence[Mapping[str, object]] | None,
    config: PerturbationBankConfig,
) -> dict[str, object]:
    if not rows:
        raise ValueError("branch has no trajectory rows")
    for row in rows:
        row["is_final"] = False
        row["recovery_status"] = "pending"
    if spec.branch_kind == "frozen":
        status = "frozen"
        summary = {
            "status": status,
            "source_return_supported": False,
            "recovery_fraction": 0.0,
            "source_label_retention": sum(
                float(row.get("source_label_retention", 0.0) or 0.0) for row in rows
            )
            / len(rows),
            "frozen_control_available": True,
        }
    elif spec.branch_kind == "sham":
        status = "sham_control"
        summary = {
            "status": status,
            "source_return_supported": False,
            "recovery_fraction": None,
            "source_label_retention": sum(
                float(row.get("source_label_retention", 0.0) or 0.0) for row in rows
            )
            / len(rows),
            "frozen_control_available": False,
        }
    elif opposite_metric is None:
        status = "unassessed_no_opposite_centroid"
        summary = {
            "status": status,
            "source_return_supported": False,
            "recovery_fraction": None,
            "source_label_retention": None,
            "frozen_control_available": frozen_rows is not None,
        }
    else:
        trajectory = [
            {
                "c_on": row["c_on"],
                "c_off": row["c_off"],
                "goal": row.get("goal", 0.0) or 0.0,
                "gate": row.get("gate", 0.0) or 0.0,
            }
            for row in rows
        ]
        frozen = None
        if frozen_rows:
            frozen = [
                {
                    "c_on": row["c_on"],
                    "c_off": row["c_off"],
                    "goal": row.get("goal", 0.0) or 0.0,
                    "gate": row.get("gate", 0.0) or 0.0,
                }
                for row in frozen_rows
            ]
        try:
            recovery = classify_recovery(
                trajectory,
                spec.source_mode,
                source_metric,
                opposite_metric,
                frozen_trajectory=frozen,
                scales=_metric_scale_payload(config.metric_scales),
                minimum_source_persistence=config.minimum_source_retention,
            )
            summary = recovery.as_dict()
            if recovery.attraction_evidence:
                status = "recovered"
            elif recovery.persistent_intermediate:
                status = "persistent_intermediate"
            elif recovery.final_mode == spec.source_mode:
                status = "source_label_retained"
            else:
                status = "drifted"
            summary["status"] = status
        except (KeyError, TypeError, ValueError):
            status = "unassessed_invalid_metric_space"
            summary = {
                "status": status,
                "source_return_supported": False,
                "recovery_fraction": None,
                "source_label_retention": None,
                "frozen_control_available": frozen_rows is not None,
            }
    final = rows[-1]
    final["is_final"] = True
    final["recovery_status"] = status
    final["source_label_retention"] = summary.get("source_label_retention")
    final["recovery_fraction"] = summary.get("recovery_fraction_final", summary.get("recovery_fraction"))
    final["source_return_supported"] = summary.get("source_return_supported", False)
    diagnostics = rows[0].get("intervention_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "source_run_id": spec.source_run_id,
        "source_step": spec.source_step,
        "source_label": spec.source_mode,
        "branch_id": spec.branch_id,
        "branch_kind": spec.branch_kind,
        "control_kind": spec.control_kind,
        "intervention": spec.intervention,
        "strength": spec.strength,
        "branch_seed": spec.branch_seed,
        "status": status,
        "recovery_status": status,
        "source_return_supported": summary.get("source_return_supported", False),
        "recovery_fraction": summary.get("recovery_fraction_final", summary.get("recovery_fraction")),
        "source_distance_initial": summary.get("source_distance_initial", summary.get("initial_source_distance")),
        "source_distance_final": summary.get("source_distance_final", summary.get("final_source_distance")),
        "opposite_distance_final": summary.get("opposite_distance_final"),
        "source_label_retention": summary.get("source_label_retention"),
        "frozen_control_available": summary.get("frozen_control_available", frozen_rows is not None),
        "dynamic_pull_final": summary.get("late_dynamic_pull", summary.get("dynamic_pull_final")),
        "intervention_feasible": all(
            bool(row.get("intervention_feasible", True)) for row in rows
        ),
        "intervention_status": rows[0].get("intervention_status", "applied"),
        "parameter_delta_norm": diagnostics.get("parameter_delta_norm"),
        "parameter_relative_delta": diagnostics.get("parameter_relative_delta"),
        "gradient_norm": diagnostics.get("gradient_norm"),
        "update_norm": diagnostics.get("update_norm"),
        "intervention_diagnostics": dict(diagnostics),
        "source_checkpoint_hash": spec.source_checkpoint_hash,
        "source_checkpoint_digest": spec.source_checkpoint_hash,
        "source_rng_digest": spec.source_rng_digest,
        "data_fingerprint": spec.data_fingerprint,
        "eval_fingerprint": spec.eval_fingerprint,
    }


def run_perturbation_bank(
    *,
    session_dir: str | os.PathLike[str],
    sources: Sequence[SourceSnapshot | Mapping[str, object]],
    train_episodes: Sequence[object],
    eval_pairs: Sequence[PairedEpisode],
    values: Mapping[str, object] | PerturbationBankConfig,
    config_identity: str,
    source_identity: str,
    dataset_hash: str,
    eval_hash: str,
    model_factory: ModelFactory | None = None,
    interventions: Sequence[tuple[str, object]] | Mapping[str, Sequence[object]] | None = None,
    evaluate_fn: MetricEvaluator | None = None,
    evaluate_bank_fn: BankMetricEvaluator | None = None,
    thresholds: ModeThresholds | None = None,
) -> PerturbationBankResult:
    """Run or resume all perturbation groups.

    ``model_factory`` is the one integration hook required when a source
    record carries tensors but not the scalar module class.  ``evaluate_bank_fn``
    can provide a vectorised evaluator such as ``bank_campaign._evaluate_bank``;
    the fallback materialises one replica at a time and is useful for smoke
    checks.  The continuation uses :class:`ReplicaBank` with per-row sampler
    states and grouped atomic checkpoints.
    """

    config = values if isinstance(values, PerturbationBankConfig) else PerturbationBankConfig.from_values(values)
    source_values = _coerce_sources(sources)
    if config.metric_scales is None:
        config = replace(config, metric_scales=_default_source_scale(source_values))
    source_map = {source.source_run_id: source for source in source_values}
    if len(source_map) != len(source_values):
        raise ValueError("source_run_id values must be unique")
    if evaluate_fn is None:
        evaluate_fn = lambda agent, pairs: evaluate_agent(agent, pairs, thresholds=thresholds)
    threshold_values = thresholds or ModeThresholds()
    specs = build_branch_specs(
        source_values,
        interventions=interventions,
        resume_steps=config.resume_steps,
        data_fingerprint=dataset_hash,
        eval_fingerprint=eval_hash,
    )
    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[PerturbationBranchSpec]] = {}
    for spec in specs:
        groups.setdefault(_group_key(spec), []).append(spec)
    trajectory_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    lineages: dict[str, dict[str, object]] = {}
    group_ids: list[str] = []
    frozen_by_key: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {}

    for group_key, group_values in sorted(groups.items()):
        group_specs = tuple(group_values)
        group_id = _group_id(group_specs, config_identity, source_identity)
        group_ids.append(group_id)
        group_store = CheckpointStore(
            root / "groups",
            group_id,
            config_identity=config_identity,
            source_identity=source_identity,
        )
        manifest = _group_manifest(
            group_id,
            group_specs,
            config=config,
            config_identity=config_identity,
            source_identity=source_identity,
            dataset_hash=dataset_hash,
            eval_hash=eval_hash,
        )
        _validate_group_manifest(group_store.run_dir / "manifest.json", manifest)
        lineage_path = group_store.run_dir / "raw" / "lineage.json"
        lineage_payload = {
            spec.branch_id: spec.lineage() for spec in group_specs
        }
        if lineage_path.is_file():
            existing_lineage = _read_json(lineage_path)
            if existing_lineage != lineage_payload:
                # Existing records may contain post/final hashes.  Identity
                # fields must match while mutable diagnostics are preserved.
                for branch_id, expected in lineage_payload.items():
                    previous = existing_lineage.get(branch_id)
                    if not isinstance(previous, Mapping):
                        raise ValueError(f"lineage is missing branch {branch_id}")
                    for key in ("branch_id", "source_run_id", "source_checkpoint", "intervention", "strength", "branch_kind", "control_kind"):
                        if previous.get(key) != expected.get(key):
                            raise ValueError(f"lineage identity mismatch for {branch_id}")
                lineage_payload = {str(key): dict(value) for key, value in existing_lineage.items() if isinstance(value, Mapping)}
        else:
            _write_json(lineage_path, lineage_payload)
        lineages.update({key: dict(value) for key, value in lineage_payload.items()})
        trajectory_path = group_store.run_dir / "raw" / "trajectory.jsonl"
        current_rows = _read_jsonl(trajectory_path)
        by_branch_horizon = {
            (str(row.get("branch_id")), int(row.get("step_since_branch", -1))): row
            for row in current_rows
        }
        latest_payload: Mapping[str, object] | None = None
        resume_rng: Mapping[str, object] | None = None
        latest_ref = group_store.latest()
        if latest_ref is not None:
            loaded_group = group_store.load(
                latest_ref,
                load_torch=True,
                expected_config_identity=config_identity,
            )
            latest_payload = loaded_group.torch_state
            if not isinstance(latest_payload, Mapping):
                raise CheckpointError("group checkpoint has no torch payload")
            if int(latest_payload.get("schema_version", -1)) != PERTURBATION_BANK_SCHEMA_VERSION:
                raise CheckpointError("unsupported perturbation bank checkpoint schema")
            resume_rng_values: dict[str, object] = dict(loaded_group.rng_state or {})
            torch_rng = latest_payload.get("torch_rng")
            if isinstance(torch_rng, Mapping):
                resume_rng_values.update(torch_rng)
            resume_rng = resume_rng_values
            if latest_payload.get("dataset_sha256") != dataset_hash or latest_payload.get("evaluation_sha256") != eval_hash:
                raise ValueError("group checkpoint data identity mismatch")
        bank, models, optimizers, diagnostics, state = _prepare_group(
            group_specs,
            source_map,
            model_factory=model_factory,
            config=config,
            evaluate_fn=evaluate_fn,
            pairs=eval_pairs,
            resume_payload=latest_payload,
        )
        del models, optimizers
        for index, spec in enumerate(group_specs):
            post_hash = state_fingerprint(bank.replica_state_dict(index))
            lineages.setdefault(spec.branch_id, {}).setdefault(
                "post_intervention_checkpoint_hash", post_hash
            )
            lineages[spec.branch_id].setdefault(
                "parent_checkpoint_hash", spec.source_checkpoint_hash
            )
            lineages[spec.branch_id].setdefault(
                "intervention_diagnostics", diagnostics[index]
            )
        if group_store.is_run_complete() and latest_payload is not None:
            state = _make_bank_state(
                [
                    latest_payload["branches"][spec.branch_id]["training_state"]  # type: ignore[index]
                    for spec in group_specs
                ],
                replica_count=len(group_specs),
                source_step=group_specs[0].source_step,
            )

        source_horizons = set(int(row["step_since_branch"]) for row in by_branch_horizon.values())
        completed_horizon = max(source_horizons, default=-1)

        def append_evaluated_rows(
            metrics: Sequence[Mapping[str, object]],
            horizon: int,
            train_values: Sequence[Mapping[str, object]] | None = None,
        ) -> None:
            for index, spec in enumerate(group_specs):
                source = source_map[spec.source_run_id]
                metric_value = dict(metrics[index])
                metric_value.setdefault(
                    "n_pairs",
                    sum(bool(pair.on.secret_opportunity) for pair in eval_pairs),
                )
                train_value = {} if train_values is None else train_values[index]
                row = _branch_metric_row(
                    metric_value,
                    spec=spec,
                    source_metric=source.source_metric,
                    opposite_metric=_source_opposite(source_map, source),
                    step=spec.source_step + horizon,
                    horizon=horizon,
                    eval_hash=eval_hash,
                    train_reward=train_value.get("expected_reward"),
                    train_loss=train_value.get("loss"),
                    thresholds=threshold_values,
                    intervention_diagnostics=diagnostics[index],
                    metric_scales=config.metric_scales,
                )
                by_branch_horizon[(spec.branch_id, horizon)] = row
            _write_jsonl(
                trajectory_path,
                [by_branch_horizon[key] for key in sorted(by_branch_horizon)],
            )

        if group_specs[0].branch_kind == "frozen":
            target_horizons = config.horizons
            metrics = _call_bank_evaluator(
                evaluate_bank_fn,
                bank,
                group_specs,
                eval_pairs,
                fallback=evaluate_fn,
            )
            for horizon in target_horizons:
                if horizon <= completed_horizon:
                    continue
                append_evaluated_rows(metrics, horizon)
            if not group_store.is_run_complete():
                # Frozen controls still get a marker bound to a checkpoint so
                # restart detection remains identical to resumed groups.
                save_group_checkpoint(
                    group_store,
                    bank=bank,
                    state=state,
                    specs=group_specs,
                    diagnostics=diagnostics,
                    group_id=group_id,
                    config=config,
                    config_identity=config_identity,
                    source_identity=source_identity,
                    dataset_hash=dataset_hash,
                    eval_hash=eval_hash,
                    metrics=metrics,
                )
                group_store.mark_run_complete(summary={"final_horizon": config.horizons[-1], "branch_kind": "frozen"})
        else:
            # Record the post-intervention point before any continuation.  If
            # a worker died after a grouped checkpoint was committed, recover
            # its missing trajectory row from that checkpoint state as well.
            current_horizon = max(0, int(state.global_step) - group_specs[0].source_step)
            if 0 not in source_horizons:
                append_evaluated_rows(
                    _call_bank_evaluator(
                        evaluate_bank_fn,
                        bank,
                        group_specs,
                        eval_pairs,
                        fallback=evaluate_fn,
                    ),
                    0,
                )
                source_horizons.add(0)
            if current_horizon > completed_horizon and current_horizon > 0:
                recovered_metrics = _call_bank_evaluator(
                    evaluate_bank_fn,
                    bank,
                    group_specs,
                    eval_pairs,
                    fallback=evaluate_fn,
                )
                recovered_training = []
                if latest_payload is not None and isinstance(latest_payload.get("metrics"), Sequence):
                    for item in latest_payload["metrics"]:
                        recovered_training.append(item if isinstance(item, Mapping) else {})
                if len(recovered_training) != len(group_specs):
                    recovered_training = [{} for _ in group_specs]
                append_evaluated_rows(recovered_metrics, current_horizon, recovered_training)
                completed_horizon = current_horizon
            for horizon in config.horizons:
                if horizon <= completed_horizon:
                    continue
                current_step = int(state.global_step)
                target_step = spec.source_step + horizon
                if target_step <= current_step:
                    continue
                interval = max(1, target_step - current_step)
                training_config = {
                    "steps": target_step,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "kl_coefficient": config.kl_coefficient,
                    "grad_clip_norm": config.grad_clip_norm,
                    "checkpoint_every_steps": interval,
                    "device": config.device,
                    "execution": config.execution,
                    "replicas": len(group_specs),
                }
                callback_metrics: list[dict[str, object]] = []

                def callback(*, bank: ReplicaBank, state: BatchedTrainingState, metrics: Mapping[str, object], **_: object) -> None:
                    nonlocal callback_metrics
                    callback_metrics = [
                        {
                            str(key): _replica_value(value, index)
                            for key, value in metrics.items()
                        }
                        for index in range(len(group_specs))
                    ]
                    save_group_checkpoint(
                        group_store,
                        bank=bank,
                        state=state,
                        specs=group_specs,
                        diagnostics=diagnostics,
                        group_id=group_id,
                        config=config,
                        config_identity=config_identity,
                        source_identity=source_identity,
                        dataset_hash=dataset_hash,
                        eval_hash=eval_hash,
                        metrics=callback_metrics,
                    )

                if latest_payload is not None and state.global_step > current_step:
                    latest_payload = None
                # The scalar runner restores the source RNG before resuming
                # its sampler/optimizer.  ReplicaBank uses explicit sampler
                # states, while restoring here also preserves any torch or
                # NumPy stream a caller's model may consume.
                stream_rng = resume_rng or source_map[group_specs[0].source_run_id].rng_state
                if stream_rng:
                    _restore_rng(stream_rng)
                state = bank.train(
                    collate(tuple(train_episodes)),
                    training_config,
                    config.reward_config or RewardConfig(),
                    state=state,
                    checkpoint_callback=callback,
                )
                resume_rng = _capture_rng()
                metrics = _call_bank_evaluator(
                    evaluate_bank_fn,
                    bank,
                    group_specs,
                    eval_pairs,
                    fallback=evaluate_fn,
                )
                append_evaluated_rows(metrics, horizon, callback_metrics or [{} for _ in group_specs])
                for index, spec in enumerate(group_specs):
                    final_hash = state_fingerprint(bank.replica_state_dict(index))
                    lineages[spec.branch_id]["final_checkpoint_hash"] = final_hash
                completed_horizon = horizon
            if not group_store.is_run_complete():
                group_store.mark_run_complete(
                    summary={
                        "final_horizon": config.horizons[-1],
                        "branch_kind": group_specs[0].branch_kind,
                        "branch_ids": [spec.branch_id for spec in group_specs],
                    }
                )
        _write_json(
            lineage_path,
            {spec.branch_id: lineages[spec.branch_id] for spec in group_specs},
        )
        final_group_rows: dict[str, list[dict[str, object]]] = {}
        for row in by_branch_horizon.values():
            final_group_rows.setdefault(str(row.get("branch_id")), []).append(dict(row))
        for spec in group_specs:
            source = source_map[spec.source_run_id]
            source_final_rows = sorted(final_group_rows.get(spec.branch_id, ()), key=lambda row: int(row.get("step_since_branch", 0)))
            frozen_rows = frozen_by_key.get((spec.intervention, spec.source_run_id), {}).get("rows")
            summary = _set_recovery_fields(
                source_final_rows,
                spec=spec,
                source_metric=source.source_metric,
                opposite_metric=_source_opposite(source_map, source),
                frozen_rows=frozen_rows,
                config=config,
            )
            if spec.branch_kind == "frozen":
                frozen_by_key.setdefault((spec.intervention, spec.source_run_id), {})["rows"] = source_final_rows
            trajectory_rows.extend(source_final_rows)
            summaries.append(summary)
    for row in trajectory_rows:
        row["is_final"] = bool(int(row.get("step_since_branch", -1)) == config.horizons[-1])
    _write_jsonl(root / "perturbation_trajectory.jsonl", trajectory_rows)
    _write_json(root / "perturbation_summary.json", summaries)
    return PerturbationBankResult(
        tables={
            "perturbation_trajectory.csv": tuple(trajectory_rows),
            "perturbation_summary.csv": tuple(summaries),
        },
        branch_specs=tuple(specs),
        group_ids=tuple(group_ids),
        lineages={key: dict(value) for key, value in lineages.items()},
    )


# Compatibility aliases for notebook code and hidden integration tests.
PerturbationSpec = PerturbationBranchSpec
BranchSpec = PerturbationBranchSpec
PerturbationBank = PerturbationBankResult
run_branches = run_perturbation_bank
run_perturbation_branches = run_perturbation_bank
build_perturbation_specs = build_branch_specs
group_specs = group_branch_specs
apply_perturbation = apply_intervention


__all__ = [
    "BranchSpec",
    "DEFAULT_HORIZONS",
    "DEFAULT_INTERVENTIONS",
    "PERTURBATION_BANK_SCHEMA_VERSION",
    "PerturbationBank",
    "PerturbationBankConfig",
    "PerturbationBankResult",
    "PerturbationBranchSpec",
    "PerturbationSpec",
    "SourceSnapshot",
    "apply_intervention",
    "apply_perturbation",
    "build_branch_specs",
    "build_perturbation_specs",
    "group_branch_specs",
    "group_specs",
    "load_group_checkpoint",
    "load_source_snapshot",
    "run_branches",
    "run_perturbation_branches",
    "run_perturbation_bank",
    "save_group_checkpoint",
    "state_fingerprint",
]
