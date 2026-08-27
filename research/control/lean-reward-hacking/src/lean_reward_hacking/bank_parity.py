"""Fail-closed parity checks for the vectorised replica bank.

The campaign runner uses :class:`~lean_reward_hacking.batched_training.ReplicaBank`
to keep many independent toy replicas in one leading-dimension tensor.  This
module is the small numerical gate that justifies that substitution.  It runs
the registered scalar objective and the bank objective side by side, with the
same initial parameters, sampler streams, minibatches, and Adam settings.

The gate is intentionally Colab-only.  It is a validation of the runtime used
by the real sweep, not a local fallback.  Its output is a compact JSON record;
bank checkpoints and the torch payload used for the resume check stay under
the supplied remote root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .checkpoints import CheckpointStore
from .config import ExperimentConfig, canonical_config, config_hash, load_config
from .episodes import dataset_fingerprint, make_paired_evaluation, make_training_episodes
from .evaluation import ModeThresholds, evaluate_agent
from .provenance import atomic_write_json, collect_provenance, hash_tree, sha256_file
from .safety import ContractViolation, assert_colab_execution

try:  # PyTorch is a pinned Colab dependency and optional for local imports.
    import torch

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local path
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


PARITY_SCHEMA_VERSION = "lrh-bank-parity/v1"
PARITY_DONE_NAME = "parity.done.json"
PARITY_REPORT_NAME = "parity.json"
PARITY_CHECKPOINT_RUN_ID = "bank-resume"

# These tolerances are part of the validation contract and are written into
# every report.  CPU is expected to be bitwise close.  CUDA can choose a
# different reduction order for the vectorised and scalar paths, so the gate
# permits the documented small floating-point envelope.
PARITY_TOLERANCES: dict[str, dict[str, float]] = {
    "cpu": {
        "logits_atol": 1.0e-6,
        "logits_rtol": 1.0e-5,
        "loss_atol": 1.0e-6,
        "loss_rtol": 1.0e-5,
        "grad_atol": 2.0e-6,
        "grad_rtol": 2.0e-5,
        "clipping_atol": 2.0e-6,
        "clipping_rtol": 2.0e-5,
        "adam_atol": 3.0e-6,
        "adam_rtol": 3.0e-5,
        "parameters_atol": 3.0e-6,
        "parameters_rtol": 3.0e-5,
        "metrics_atol": 1.0e-6,
        "metrics_rtol": 1.0e-5,
    },
    "cuda": {
        "logits_atol": 2.0e-5,
        "logits_rtol": 2.0e-4,
        "loss_atol": 2.0e-5,
        "loss_rtol": 2.0e-4,
        "grad_atol": 4.0e-5,
        "grad_rtol": 4.0e-4,
        "clipping_atol": 4.0e-5,
        "clipping_rtol": 4.0e-4,
        "adam_atol": 5.0e-5,
        "adam_rtol": 5.0e-4,
        "parameters_atol": 5.0e-5,
        "parameters_rtol": 5.0e-4,
        "metrics_atol": 2.0e-5,
        "metrics_rtol": 2.0e-4,
    },
}


class BankParityError(RuntimeError):
    """Raised when scalar and bank execution disagree or provenance is invalid."""


@dataclass(frozen=True, slots=True)
class ParityResult:
    """JSON-friendly result returned by :func:`run_bank_parity`."""

    status: str
    report_path: str
    done_path: str
    report: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "report_path": self.report_path,
            "done_path": self.done_path,
            "report": self.report,
        }

    as_dict = to_dict


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise BankParityError(
            "bank parity requires the pinned PyTorch Colab runtime; local execution is disabled"
        )


def _as_mapping(config: ExperimentConfig | Mapping[str, object] | str | os.PathLike[str] | None) -> tuple[dict[str, object], str, str | None]:
    """Return flat config values, identity hash, and source path when present."""

    if config is None:
        values: dict[str, object] = {
            "schema_version": 1,
            "experiment": "bank_parity",
            "execution": "colab",
            "task_dim": 4,
            "hidden_width": 4,
            "harmful_goal_strength": 2.0,
            "initial_audit_sensitivity": 0.0,
            "learning_rate": 0.003,
            "weight_decay": 1.0e-4,
            "entropy_coefficient": 0.02,
            "grad_clip_norm": 1.0,
        }
        return values, config_hash(values), None
    if isinstance(config, (str, os.PathLike)):
        path = Path(config).resolve()
        loaded = load_config(path)
        return dict(loaded.values), config_hash(loaded), str(path)
    if isinstance(config, ExperimentConfig):
        return dict(config.values), config_hash(config), None
    if isinstance(config, Mapping):
        values = dict(config)
        return values, config_hash(values), None
    raise TypeError("config must be a path, ExperimentConfig, mapping, or None")


def _numeric(value: object, name: str, default: float) -> float:
    supplied = default if value is None else value
    if isinstance(supplied, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(supplied)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str, default: int) -> int:
    supplied = default if value is None else value
    if isinstance(supplied, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(supplied)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _device_name(requested: object | None, values: Mapping[str, object]) -> str:
    value = str(values.get("device", "cuda") if requested is None else requested).strip().lower()
    if value in {"gpu", "cuda:0"}:
        return "cuda"
    if value in {"", "cpu"}:
        return "cpu"
    if value.startswith("cuda"):
        return "cuda"
    raise ValueError(f"bank parity device must be cpu or cuda, got {value!r}")


def _architecture_name(requested: object | None, values: Mapping[str, object]) -> str:
    if requested is not None:
        value = str(requested)
    else:
        value = str(values.get("architecture", values.get("model_kind", "toy")))
        if value == "toy" and str(values.get("experiment", "")).lower() == "generic_mlp":
            value = "generic"
    value = value.strip().lower().replace("_", "-")
    if value in {"toy", "goal-gate", "goalgate"}:
        return "toy"
    if value in {"generic", "mlp", "plain-mlp", "plain", "generic-mlp"}:
        return "generic"
    raise ValueError("bank parity architecture must be toy or generic")


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_provenance(
    *,
    values: Mapping[str, object],
    config_identity: str,
    source_identity: str,
    seeds: Sequence[int],
    device: str,
    architecture: str,
) -> dict[str, object]:
    packages = ("torch", "numpy")
    return collect_provenance(
        _source_root(),
        config={"values": dict(values), "config_sha256": config_identity},
        seeds=seeds,
        package_names=packages,
        extra={
            "source_archive_sha256": source_identity,
            "device": device,
            "architecture": architecture,
            "parity_schema_version": PARITY_SCHEMA_VERSION,
        },
    )


def _feature_tensor(episodes: Sequence[object], device: str) -> "torch.Tensor":
    _require_torch()
    rows: list[list[float]] = []
    for episode in episodes:
        rows.append(
            [
                *[float(value) for value in episode.task],
                float(bool(episode.audit)),
                float(bool(episode.secret_opportunity)),
            ]
        )
    if not rows:
        raise ValueError("parity training data cannot be empty")
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _model(
    architecture: str,
    values: Mapping[str, object],
    *,
    seed: int,
    task_dim: int,
) -> "torch.nn.Module":
    _require_torch()
    if architecture == "generic":
        from .generic import initialize_generic_agent

        return initialize_generic_agent(
            task_dim=task_dim,
            hidden_width=_positive_int(values.get("hidden_width"), "hidden_width", 4),
            harmful_goal_strength=_numeric(
                values.get("harmful_goal_strength", values.get("initial_harmful_goal_strength")),
                "harmful_goal_strength",
                2.0,
            ),
            model_seed=seed,
            device="cpu",
        )
    from .toy import initialize_toy_agent

    return initialize_toy_agent(
        task_dim=task_dim,
        hidden_width=_positive_int(values.get("hidden_width"), "hidden_width", 4),
        harmful_goal_strength=_numeric(
            values.get("harmful_goal_strength", values.get("initial_harmful_goal_strength")),
            "harmful_goal_strength",
            2.0,
        ),
        initial_audit_sensitivity=_numeric(
            values.get("initial_audit_sensitivity", values.get("audit_sensitivity")),
            "initial_audit_sensitivity",
            0.0,
        ),
        model_seed=seed,
        device="cpu",
    )


def _reward_coefficients(values: Mapping[str, object]) -> tuple[float, float, float, float | None, float]:
    help_reward = _numeric(values.get("help_reward"), "help_reward", 1.0)
    harm_reward = _numeric(values.get("harm_reward"), "harm_reward", -1.0)
    kl = _numeric(
        values.get("kl_coefficient", values.get("entropy_coefficient")),
        "kl_coefficient",
        0.02,
    )
    l2 = _numeric(values.get("weight_decay"), "weight_decay", 1.0e-4)
    clip_value = values.get("grad_clip_norm", 1.0)
    clip = None if clip_value is None else _numeric(clip_value, "grad_clip_norm", 1.0)
    return help_reward, harm_reward, kl, clip, l2


def _tensor_list(value: object) -> "torch.Tensor":
    _require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device="cpu")


def _max_abs(left: object, right: object) -> float:
    a = _tensor_list(left)
    b = _tensor_list(right)
    if tuple(a.shape) != tuple(b.shape):
        raise BankParityError(f"parity shape mismatch: {tuple(a.shape)} versus {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _close(left: object, right: object, *, category: str, tolerance: Mapping[str, float]) -> float:
    """Compare values and return the maximum absolute deviation."""

    difference = _max_abs(left, right)
    a = _tensor_list(left)
    b = _tensor_list(right)
    atol = float(tolerance[f"{category}_atol"])
    rtol = float(tolerance[f"{category}_rtol"])
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise BankParityError(
            f"bank parity {category} mismatch: max_abs={difference:.9g}, atol={atol}, rtol={rtol}"
        )
    return difference


def _metric_value(value: object, index: int | None = None) -> object:
    if isinstance(value, torch.Tensor):
        if index is not None:
            value = value[index]
        return value.detach().cpu()
    return value


def _state_values(optimizer: "torch.optim.Optimizer", model: "torch.nn.Module") -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, parameter in model.named_parameters():
        result[name] = {
            str(key): value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in optimizer.state.get(parameter, {}).items()
        }
    return result


def _compare_optimizer_state(
    scalar_optimizer: "torch.optim.Optimizer",
    scalar_model: "torch.nn.Module",
    bank: object,
    replica: int,
    tolerance: Mapping[str, float],
) -> float:
    scalar_state = _state_values(scalar_optimizer, scalar_model)
    bank_state = bank.replica_optimizer_state(replica)
    if set(scalar_state) != set(bank_state):
        raise BankParityError("bank and scalar optimizer parameter names differ")
    maximum = 0.0
    for name in scalar_state:
        if set(scalar_state[name]) != set(bank_state[name]):
            raise BankParityError(f"Adam state keys differ for parameter {name!r}")
        for key in scalar_state[name]:
            left = scalar_state[name][key]
            right = bank_state[name][key]
            if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
                maximum = max(maximum, _close(left, right, category="adam", tolerance=tolerance))
            elif left != right:
                raise BankParityError(f"Adam scalar state differs for {name}.{key}")
    return maximum


def _compare_optimizer_state_maps(
    left: Mapping[str, Mapping[str, object]],
    right: Mapping[str, Mapping[str, object]],
    tolerance: Mapping[str, float],
) -> float:
    """Compare two name-keyed Adam state maps, including scalar step tensors."""

    if set(left) != set(right):
        raise BankParityError("bank optimizer parameter names differ after resume")
    maximum = 0.0
    for name in left:
        if set(left[name]) != set(right[name]):
            raise BankParityError(f"Adam state keys differ for parameter {name!r} after resume")
        for key in left[name]:
            value_left = left[name][key]
            value_right = right[name][key]
            if isinstance(value_left, torch.Tensor) or isinstance(value_right, torch.Tensor):
                maximum = max(
                    maximum,
                    _close(value_left, value_right, category="adam", tolerance=tolerance),
                )
            elif value_left != value_right:
                raise BankParityError(f"Adam scalar state differs for {name}.{key} after resume")
    return maximum


def _sampler_step(
    generators: Sequence["torch.Generator"],
    permutations: list["torch.Tensor | None"],
    *,
    n_samples: int,
    batch_size: int,
    batch_offset: int,
    epoch: int,
) -> tuple[list["torch.Tensor"], int, int, list[int]]:
    if any(permutation is None for permutation in permutations) or batch_offset >= n_samples:
        if all(permutation is not None for permutation in permutations) and batch_offset >= n_samples:
            epoch += 1
        batch_offset = 0
        for index, generator in enumerate(generators):
            permutations[index] = torch.randperm(n_samples, generator=generator, device="cpu")
    start = int(batch_offset)
    stop = min(start + int(batch_size), n_samples)
    assert all(permutation is not None for permutation in permutations)
    indices = [permutation[start:stop] for permutation in permutations]  # type: ignore[index]
    return indices, stop, epoch, [int(value) for value in indices[0].tolist()]


def _write_failure(report_path: Path, report: Mapping[str, object]) -> None:
    try:
        atomic_write_json(report_path, dict(report))
    except OSError:
        # The original numerical exception carries the useful failure detail.
        pass


def _checkpoint_resume_check(
    *,
    values: Mapping[str, object],
    architecture: str,
    task_dim: int,
    device: str,
    features: "torch.Tensor",
    remote_root: Path,
    config_identity: str,
    source_identity: str,
    checkpoint_run_id: str,
    seed: int,
    total_steps: int,
    split_step: int,
    batch_size: int,
    help_reward: float,
    harm_reward: float,
    kl: float,
    l2: float,
    clip: float | None,
    tolerance: Mapping[str, float],
) -> dict[str, object]:
    """Exercise atomic storage, cursor recovery, and scalar materialisation."""

    from .batched_training import BatchedTrainingConfig, ReplicaBank

    model_seeds = (seed + 100, seed + 101)
    sampler_seeds = (seed + 10_000, seed + 10_001)
    initial = [_model(architecture, values, seed=item, task_dim=task_dim) for item in model_seeds]
    config = BatchedTrainingConfig(
        steps=total_steps,
        batch_size=batch_size,
        learning_rate=_numeric(values.get("learning_rate"), "learning_rate", 0.003),
        weight_decay=l2,
        kl_coefficient=kl,
        grad_clip_norm=clip,
        checkpoint_every_steps=split_step,
        device=device,
    )
    bank_continuous = ReplicaBank.from_agents(
        initial,
        model_seeds=model_seeds,
        sampler_seeds=sampler_seeds,
        device=device,
        learning_rate=config.learning_rate,
    )
    continuous_state = bank_continuous.train(
        features,
        config,
        steps=total_steps,
        reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
    )

    bank_split = ReplicaBank.from_agents(
        initial,
        model_seeds=model_seeds,
        sampler_seeds=sampler_seeds,
        device=device,
        learning_rate=config.learning_rate,
    )
    split_config = BatchedTrainingConfig(
        steps=split_step,
        batch_size=batch_size,
        learning_rate=config.learning_rate,
        weight_decay=l2,
        kl_coefficient=kl,
        grad_clip_norm=clip,
        checkpoint_every_steps=split_step,
        device=device,
    )
    parity_checkpoint_root = remote_root / "parity_checkpoints"
    store = CheckpointStore(
        parity_checkpoint_root,
        checkpoint_run_id,
        config_identity=config_identity,
        source_identity=source_identity,
    )

    def save_checkpoint(*, bank: object, state: object, metrics: object, **_: object) -> None:
        payload = bank.checkpoint_payload(
            state=state,
            config=asdict(config),
            metadata={
                "parity_schema_version": PARITY_SCHEMA_VERSION,
                "dataset_sha256": hashlib.sha256(features.detach().cpu().numpy().tobytes()).hexdigest(),
                "metrics": {
                    str(name): _metric_value(value).tolist()
                    if isinstance(_metric_value(value), torch.Tensor)
                    else value
                    for name, value in dict(metrics).items()
                },
            },
        )
        store.save(
            int(state.global_step),
            {
                "global_step": int(state.global_step),
                "epoch": int(state.epoch),
                "batch_offset": int(state.batch_offset),
                "replica_count": 2,
            },
            torch_state=payload,
            metadata={
                "parity_schema_version": PARITY_SCHEMA_VERSION,
                "config_sha256": config_identity,
                "source_archive_sha256": source_identity,
            },
            minibatch_cursor=int(state.batch_offset),
        )

    existing_split = store.latest()
    if existing_split is not None and existing_split.step == split_step:
        loaded_split = store.load(
            existing_split,
            load_torch=True,
            expected_config_identity=config_identity,
        )
        if not isinstance(loaded_split.torch_state, Mapping):
            raise BankParityError("existing split checkpoint has no torch payload")
        split_state, split_metadata = bank_split.load_checkpoint_payload(
            loaded_split.torch_state,
            map_location=device,
            expected_config=asdict(config),
        )
        if split_metadata.get("parity_schema_version") != PARITY_SCHEMA_VERSION:
            raise BankParityError("existing split checkpoint has the wrong parity schema")
    else:
        split_state = bank_split.train(
            features,
            split_config,
            steps=split_step,
            reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
            checkpoint_callback=save_checkpoint,
        )
    reference = store.latest()
    if reference is None or reference.step != split_step:
        raise BankParityError("checkpoint resume gate did not commit the split checkpoint")
    loaded = store.load(reference, load_torch=True, expected_config_identity=config_identity)
    if not isinstance(loaded.torch_state, Mapping):
        raise BankParityError("checkpoint resume gate did not recover a torch payload")

    bank_resumed = ReplicaBank.from_agents(
        initial,
        model_seeds=model_seeds,
        sampler_seeds=sampler_seeds,
        device=device,
        learning_rate=config.learning_rate,
    )
    restored_state, metadata = bank_resumed.load_checkpoint_payload(
        loaded.torch_state,
        map_location=device,
        expected_config=asdict(config),
    )
    if restored_state.global_step != split_step:
        raise BankParityError("recovered bank cursor has the wrong global step")
    if metadata.get("parity_schema_version") != PARITY_SCHEMA_VERSION:
        raise BankParityError("recovered bank metadata has the wrong schema")
    resumed_state = bank_resumed.train(
        features,
        config,
        steps=total_steps,
        state=restored_state,
        reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
    )

    max_parameter = 0.0
    for name, value in bank_continuous.parameters_by_name.items():
        max_parameter = max(
            max_parameter,
            _close(
                value,
                bank_resumed.parameters_by_name[name],
                category="parameters",
                tolerance=tolerance,
            ),
        )
    max_adam = 0.0
    for replica in range(2):
        max_adam = max(
            max_adam,
            _compare_optimizer_state_maps(
                bank_continuous.replica_optimizer_state(replica),
                bank_resumed.replica_optimizer_state(replica),
                tolerance,
            ),
        )
    # Compare the bank cursor and checkpointed history.  The history contains
    # only committed checkpoint rows by design, so this is compact and exact.
    if resumed_state.global_step != continuous_state.global_step:
        raise BankParityError("resumed bank reached a different global step")
    if resumed_state.epoch != continuous_state.epoch or resumed_state.batch_offset != continuous_state.batch_offset:
        raise BankParityError("resumed bank cursor differs from uninterrupted execution")
    if resumed_state.permutations is None or continuous_state.permutations is None:
        raise BankParityError("bank cursor did not retain the final permutation")
    if not torch.equal(resumed_state.permutations, continuous_state.permutations):
        raise BankParityError("resumed bank permutation differs from uninterrupted execution")
    if len(resumed_state.sampler_states) != len(continuous_state.sampler_states):
        raise BankParityError("resumed sampler state count differs")
    for left, right in zip(resumed_state.sampler_states, continuous_state.sampler_states, strict=True):
        if not torch.equal(torch.as_tensor(left), torch.as_tensor(right)):
            raise BankParityError("resumed sampler RNG state differs from uninterrupted execution")

    # Materialise one bank slice, restore its Adam moments, and let the scalar
    # runner continue from the identical permutation/cursor.  This catches a
    # silent optimizer-state or sampler-state loss at the bank boundary.
    from .training import TrainingState, run_training

    scalar_model, scalar_optimizer = bank_split.materialize_replica_with_optimizer(0, device=device)
    scalar_state = TrainingState(
        epoch=int(split_state.epoch),
        batch_offset=int(split_state.batch_offset),
        global_step=int(split_state.global_step),
        permutation=split_state.permutations[0].detach().cpu().clone()
        if split_state.permutations is not None
        else None,
        sampler_state=split_state.sampler_states[0].detach().cpu().clone()
        if split_state.sampler_states
        else None,
        history=[],
    )
    scalar_config = type(
        "ParityScalarConfig",
        (),
        {
            "steps": total_steps,
            "batch_size": batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": l2,
            "kl_coefficient": kl,
            "grad_clip_norm": clip,
            "checkpoint_every_steps": total_steps,
            "device": device,
            "execution": "colab",
            "replicas": 1,
            "allow_local_smoke": False,
        },
    )()
    scalar_final_state = run_training(
        scalar_model,
        scalar_optimizer,
        features,
        scalar_config,
        reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
        resume_state=scalar_state,
    )
    for name, value in scalar_model.named_parameters():
        max_parameter = max(
            max_parameter,
            _close(
                value,
                bank_resumed.parameters_by_name[name][0],
                category="parameters",
                tolerance=tolerance,
            ),
        )
    max_adam = max(
        max_adam,
        _compare_optimizer_state(
            scalar_optimizer,
            scalar_model,
            bank_resumed,
            0,
            tolerance,
        ),
    )
    if scalar_final_state.global_step != total_steps:
        raise BankParityError("materialized scalar continuation reached the wrong step")
    return {
        "checkpoint_step": split_step,
        "total_steps": total_steps,
        "checkpoint_path": str(reference.path),
        "checkpoint_metadata_sha256": reference.metadata_sha256,
        "store_run_id": checkpoint_run_id,
        "resumed_history_rows": len(resumed_state.history),
        "continuous_history_rows": len(continuous_state.history),
        "materialized_scalar_final_step": int(scalar_final_state.global_step),
        "max_parameter_difference": max_parameter,
        "max_adam_difference": max_adam,
        "sampler_cursor": {
            "epoch": int(resumed_state.epoch),
            "batch_offset": int(resumed_state.batch_offset),
            "permutation_shape": list(resumed_state.permutations.shape),
            "sampler_count": len(resumed_state.sampler_states),
        },
    }


def _run_replica_comparison(
    *,
    values: Mapping[str, object],
    architecture: str,
    task_dim: int,
    device: str,
    features: "torch.Tensor",
    pairs: Sequence[object],
    replicas: int,
    steps: int,
    batch_size: int,
    seed: int,
    tolerance: Mapping[str, float],
) -> dict[str, object]:
    from .batched_training import ReplicaBank, batched_loss_terms
    from .training import loss_terms, train_step

    model_seeds = tuple(seed + 100 + index for index in range(replicas))
    sampler_seeds = tuple(seed + 10_000 + index for index in range(replicas))
    initial = [_model(architecture, values, seed=item, task_dim=task_dim) for item in model_seeds]
    scalar_models = [copy.deepcopy(model).to(device) for model in initial]
    scalar_optimizers = [
        torch.optim.Adam(
            model.parameters(),
            lr=_numeric(values.get("learning_rate"), "learning_rate", 0.003),
        )
        for model in scalar_models
    ]
    bank = ReplicaBank.from_agents(
        initial,
        model_seeds=model_seeds,
        sampler_seeds=sampler_seeds,
        device=device,
        learning_rate=_numeric(values.get("learning_rate"), "learning_rate", 0.003),
    )
    generators = [torch.Generator(device="cpu") for _ in range(replicas)]
    for generator, sampler_seed in zip(generators, sampler_seeds, strict=True):
        generator.manual_seed(int(sampler_seed))
    permutations: list[torch.Tensor | None] = [None for _ in range(replicas)]
    batch_offset = 0
    epoch = 0
    n_samples = int(features.shape[0])
    help_reward, harm_reward, kl, clip, l2 = _reward_coefficients(values)
    metrics_max: dict[str, float] = {
        "logits": 0.0,
        "loss": 0.0,
        "grad": 0.0,
        "clipping": 0.0,
        "adam": 0.0,
        "parameters": 0.0,
        "metrics": 0.0,
    }
    sampler_trace: list[dict[str, object]] = []
    endpoint_metrics: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        indices, batch_offset, epoch, first_indices = _sampler_step(
            generators,
            permutations,
            n_samples=n_samples,
            batch_size=batch_size,
            batch_offset=batch_offset,
            epoch=epoch,
        )
        bank_batch = features[torch.stack(indices, dim=0).to(device)]
        scalar_batches = [features[index.to(device)] for index in indices]

        # Pre-update logits and objective terms are compared before either
        # optimizer moves.  This isolates model evaluation from Adam.
        bank_logits = bank.logits(bank_batch)
        bank_terms = batched_loss_terms(
            bank_logits,
            help_reward=help_reward,
            harm_reward=harm_reward,
            kl_coefficient=kl,
            l2_coefficient=l2,
            parameters=bank.parameters_by_name,
        )
        scalar_logits = [model(batch) for model, batch in zip(scalar_models, scalar_batches, strict=True)]
        scalar_terms = [
            loss_terms(
                logits,
                help_reward=help_reward,
                harm_reward=harm_reward,
                kl_coefficient=kl,
                l2_coefficient=l2,
                parameters=model.parameters(),
            )
            for logits, model in zip(scalar_logits, scalar_models, strict=True)
        ]
        for index in range(replicas):
            metrics_max["logits"] = max(
                metrics_max["logits"],
                _close(scalar_logits[index], bank_logits[index], category="logits", tolerance=tolerance),
            )
            for name in ("loss", "reward_loss", "expected_reward", "kl", "l2", "help_probability"):
                metrics_max["loss"] = max(
                    metrics_max["loss"],
                    _close(scalar_terms[index][name], bank_terms[name][index], category="loss", tolerance=tolerance),
                )

        # Hooks capture the raw gradient, while the temporary optimizer.step
        # wrapper captures the gradient after clipping and immediately before
        # Adam consumes it.  The public train_step/ReplicaBank.step paths are
        # exercised directly, with no duplicate update implementation here.
        scalar_raw: list[dict[str, torch.Tensor]] = []
        scalar_clipped: list[dict[str, torch.Tensor]] = []
        scalar_metrics: list[dict[str, float]] = []
        for model, optimizer, batch in zip(scalar_models, scalar_optimizers, scalar_batches, strict=True):
            raw: dict[str, torch.Tensor] = {}
            handles = [
                parameter.register_hook(
                    lambda gradient, name=name: raw.__setitem__(name, gradient.detach().clone())
                )
                for name, parameter in model.named_parameters()
            ]
            clipped: dict[str, torch.Tensor] = {}
            original_step = optimizer.step

            def wrapped_step(*args: object, **kwargs: object) -> object:
                clipped.update(
                    {
                        name: parameter.grad.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    }
                )
                return original_step(*args, **kwargs)

            optimizer.step = wrapped_step  # type: ignore[method-assign]
            try:
                scalar_metrics.append(
                    train_step(
                        model,
                        optimizer,
                        batch,
                        reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
                        kl_coefficient=kl,
                        l2_coefficient=l2,
                        grad_clip_norm=clip,
                    )
                )
            finally:
                optimizer.step = original_step  # type: ignore[method-assign]
                for handle in handles:
                    handle.remove()
            scalar_raw.append(raw)
            scalar_clipped.append(clipped)

        bank_raw: dict[str, torch.Tensor] = {}
        bank_handles = [
            parameter.register_hook(
                lambda gradient, name=name: bank_raw.__setitem__(name, gradient.detach().clone())
            )
            for name, parameter in bank.parameters_by_name.items()
        ]
        bank_clipped: dict[str, torch.Tensor] = {}
        bank_optimizer = bank.optimizer
        bank_original_step = bank_optimizer.step

        def bank_wrapped_step(*args: object, **kwargs: object) -> object:
            bank_clipped.update(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in bank.parameters_by_name.items()
                    if parameter.grad is not None
                }
            )
            return bank_original_step(*args, **kwargs)

        bank_optimizer.step = bank_wrapped_step  # type: ignore[method-assign]
        try:
            bank_metrics_raw = bank.step(
                bank_batch,
                reward_config={"help_reward": help_reward, "harm_reward": harm_reward},
                kl_coefficient=kl,
                l2_coefficient=l2,
                grad_clip_norm=clip,
            )
        finally:
            bank_optimizer.step = bank_original_step  # type: ignore[method-assign]
            for handle in bank_handles:
                handle.remove()

        for index in range(replicas):
            for name in bank.parameters_by_name:
                metrics_max["grad"] = max(
                    metrics_max["grad"],
                    _close(
                        scalar_raw[index][name],
                        bank_raw[name][index],
                        category="grad",
                        tolerance=tolerance,
                    ),
                )
                metrics_max["clipping"] = max(
                    metrics_max["clipping"],
                    _close(
                        scalar_clipped[index][name],
                        bank_clipped[name][index],
                        category="clipping",
                        tolerance=tolerance,
                    ),
                )
            for name in ("loss", "reward_loss", "expected_reward", "kl", "l2", "help_probability", "grad_norm"):
                metrics_max["metrics"] = max(
                    metrics_max["metrics"],
                    _close(
                        scalar_metrics[index][name],
                        _metric_value(bank_metrics_raw[name], index),
                        category="metrics",
                        tolerance=tolerance,
                    ),
                )
            for name, scalar_value in scalar_models[index].named_parameters():
                metrics_max["parameters"] = max(
                    metrics_max["parameters"],
                    _close(
                        scalar_value,
                        bank.parameters_by_name[name][index],
                        category="parameters",
                        tolerance=tolerance,
                    ),
                )
            metrics_max["adam"] = max(
                metrics_max["adam"],
                _compare_optimizer_state(
                    scalar_optimizers[index],
                    scalar_models[index],
                    bank,
                    index,
                    tolerance,
                ),
            )

        sampler_trace.append(
            {
                "step": step,
                "epoch": epoch,
                "batch_offset": batch_offset,
                "batch_indices_replica_0": first_indices,
                "permutation_sha256": [
                    hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
                    for permutation in permutations
                    if permutation is not None
                ],
            }
        )

        if step == steps:
            for index, model in enumerate(scalar_models):
                metrics = evaluate_agent(
                    model,
                    pairs,
                    thresholds=ModeThresholds(),
                    checkpoint_step=step,
                )
                endpoint_metrics.append(
                    {
                        "replica": index,
                        "c_on": metrics.c_on,
                        "c_off": metrics.c_off,
                        "gap": metrics.gap,
                        "goal": metrics.goal,
                        "gate": metrics.gate,
                        "label": metrics.mode.value if metrics.mode else None,
                    }
                )

    return {
        "replicas": replicas,
        "steps": steps,
        "crossed_epoch_boundary": epoch >= 1,
        "final_epoch": epoch,
        "final_batch_offset": batch_offset,
        "metrics_max_abs_difference": metrics_max,
        "sampler_trace": sampler_trace,
        "endpoint_metrics": endpoint_metrics,
        "model_seeds": list(model_seeds),
        "sampler_seeds": list(sampler_seeds),
    }


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _existing_result(
    done_path: Path,
    report_path: Path,
    *,
    config_identity: str,
    source_identity: str,
) -> ParityResult | None:
    marker = _read_json(done_path)
    report = _read_json(report_path)
    if marker is None or report is None:
        return None
    if marker.get("status") != "passed" or report.get("status") != "passed":
        return None
    if marker.get("config_sha256") != config_identity or marker.get("source_tree_sha256") != source_identity:
        return None
    if report.get("config_sha256") != config_identity or report.get("source_tree_sha256") != source_identity:
        return None
    return ParityResult("passed", str(report_path), str(done_path), report)


def _checkpoint_run_id(config_identity: str, source_identity: str) -> str:
    """Bind the remote checkpoint namespace to the exact validation inputs."""

    return f"{PARITY_CHECKPOINT_RUN_ID}-{config_identity[:12]}-{source_identity[:12]}"


def run_bank_parity(
    config: ExperimentConfig | Mapping[str, object] | str | os.PathLike[str] | None = None,
    remote_root: str | os.PathLike[str] | None = None,
    *,
    device: str | None = None,
    architecture: str | None = None,
    steps: int = 5,
    samples: int = 7,
    batch_size: int = 3,
    eval_pairs: int = 8,
    seed: int = 20_260_826,
    replicas: Sequence[int] = (1, 2),
) -> dict[str, object]:
    """Run the Colab-only scalar-versus-bank parity gate.

    ``steps=5, samples=7, batch_size=3`` crosses an epoch boundary while
    remaining a tiny validation job.  ``replicas`` normally contains ``1``
    and ``2``; accepting a sequence keeps focused Colab diagnostics explicit.
    """

    if remote_root is None:
        raise ValueError("remote_root is required so parity artifacts remain remote")
    values, config_identity, config_path = _as_mapping(config)
    requested_device = _device_name(device, values)
    architecture_name = _architecture_name(architecture, values)
    assert_colab_execution(require_gpu=requested_device == "cuda")
    _require_torch()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ContractViolation("bank parity requested CUDA but no Colab CUDA device is visible")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    steps = _positive_int(steps, "steps", 5)
    samples = _positive_int(samples, "samples", 7)
    batch_size = _positive_int(batch_size, "batch_size", 3)
    eval_pairs = _positive_int(eval_pairs, "eval_pairs", 8)
    replica_values = tuple(int(value) for value in replicas)
    if replica_values != (1, 2):
        raise ValueError("the registered bank parity gate compares exactly R=1 and R=2")
    task_dim = _positive_int(values.get("task_dim"), "task_dim", 4)
    source_root = _source_root()
    source_identity = hash_tree(source_root)
    root = Path(remote_root).resolve()
    output_dir = root / "parity"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / PARITY_REPORT_NAME
    done_path = output_dir / PARITY_DONE_NAME
    existing = _existing_result(
        done_path,
        report_path,
        config_identity=config_identity,
        source_identity=source_identity,
    )
    if existing is not None:
        return existing.to_dict()

    # CPU threads remain bounded even on a CUDA runtime because checkpoint
    # serialization and dataset staging use host threads.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[variable] = "2"
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError):
        # The report records the effective flag below.  A pinned runtime that
        # lacks this switch still has deterministic model/data operations.
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    started = time.monotonic()
    train_episodes = make_training_episodes(
        samples,
        task_dim,
        seed + 1,
        opportunity_probability=0.5,
    )
    pairs = make_paired_evaluation(
        eval_pairs,
        task_dim,
        seed + 2,
        opportunity_probability=1.0,
    )
    features = _feature_tensor(train_episodes, requested_device)
    help_reward, harm_reward, kl, clip, l2 = _reward_coefficients(values)
    tolerance = PARITY_TOLERANCES["cuda" if requested_device == "cuda" else "cpu"]
    checkpoint_run_id = _checkpoint_run_id(config_identity, source_identity)
    provenance = _runtime_provenance(
        values=values,
        config_identity=config_identity,
        source_identity=source_identity,
        seeds=(seed, seed + 1, seed + 2),
        device=requested_device,
        architecture=architecture_name,
    )
    report: dict[str, object] = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "status": "running",
        "config_sha256": config_identity,
        "config_path": config_path,
        "config": dict(values),
        "source_tree_sha256": source_identity,
        "architecture": architecture_name,
        "device": requested_device,
        "steps": steps,
        "samples": samples,
        "batch_size": batch_size,
        "eval_pairs": eval_pairs,
        "seeds": {
            "parity_seed": seed,
            "dataset_seed": seed + 1,
            "evaluation_seed": seed + 2,
        },
        "dataset_sha256": dataset_fingerprint(train_episodes),
        "evaluation_sha256": dataset_fingerprint(pairs),
        "checkpoint_run_id": checkpoint_run_id,
        "objective": {
            "help_reward": help_reward,
            "harm_reward": harm_reward,
            "kl_coefficient": kl,
            "l2_coefficient": l2,
            "grad_clip_norm": clip,
            "optimizer": "Adam",
        },
        "tolerances": {"device_class": "cuda" if requested_device == "cuda" else "cpu", **tolerance},
        "provenance": provenance,
    }
    atomic_write_json(report_path, report)
    try:
        comparisons: dict[str, object] = {}
        for replica_count in (1, 2):
            comparisons[f"R={replica_count}"] = _run_replica_comparison(
                values=values,
                architecture=architecture_name,
                task_dim=task_dim,
                device=requested_device,
                features=features,
                pairs=pairs,
                replicas=replica_count,
                steps=steps,
                batch_size=batch_size,
                seed=seed,
                tolerance=tolerance,
            )
        resume = _checkpoint_resume_check(
            values=values,
            architecture=architecture_name,
            task_dim=task_dim,
            device=requested_device,
            features=features,
            remote_root=root,
            config_identity=config_identity,
            source_identity=source_identity,
            checkpoint_run_id=checkpoint_run_id,
            seed=seed,
            total_steps=max(steps, 5),
            split_step=max(2, min(steps - 1, 3)),
            batch_size=batch_size,
            help_reward=help_reward,
            harm_reward=harm_reward,
            kl=kl,
            l2=l2,
            clip=clip,
            tolerance=tolerance,
        )
        report.update(
            {
                "status": "passed",
                "comparisons": comparisons,
                "checkpoint_resume": resume,
                "elapsed_seconds": time.monotonic() - started,
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }
        )
        atomic_write_json(report_path, report)
        marker = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "status": "passed",
            "config_sha256": config_identity,
            "source_tree_sha256": source_identity,
            "report_sha256": sha256_file(report_path),
            "checkpoint_run_id": checkpoint_run_id,
            "device": requested_device,
            "architecture": architecture_name,
            "steps": steps,
        }
        atomic_write_json(done_path, marker)
        return ParityResult("passed", str(report_path), str(done_path), report).to_dict()
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _write_failure(report_path, report)
        raise


run_parity = run_bank_parity
bank_parity_gate = run_bank_parity


__all__ = [
    "BankParityError",
    "PARITY_DONE_NAME",
    "PARITY_REPORT_NAME",
    "PARITY_SCHEMA_VERSION",
    "PARITY_TOLERANCES",
    "ParityResult",
    "TORCH_AVAILABLE",
    "bank_parity_gate",
    "run_bank_parity",
    "run_parity",
]
