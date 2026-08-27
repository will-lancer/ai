"""Restartable vectorised execution for the toy initial-condition map.

The basin scan changes only the registered initial conditions.  Replicas in a
single phase bank share the exact audited training data, loss, optimizer, and
checkpoint cadence.  Their initial model parameters and sampler streams are
independent leading-dimension slices of :class:`ReplicaBank`.

The raw checkpoint JSONL and bank checkpoints belong in the Colab/Drive
session directory.  The returned tables contain only compact rows suitable
for the normal campaign exporter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import json
from pathlib import Path

from .batched_training import BatchedTrainingState, ReplicaBank
from .checkpoints import CheckpointStore
from .episodes import dataset_fingerprint
from .evaluation import EvaluationMetrics, ModeThresholds
from .rewards import RewardConfig
from .types import Mode, PairedEpisode


BASIN_BANK_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BasinReplicaSpec:
    """The complete deterministic identity of one basin-map replica."""

    phase: str
    level: int
    harm_index: int
    audit_index: int
    seed_index: int
    harm_strength: float
    audit_sensitivity: float
    model_seed: int
    sampler_seed: int
    run_id: str
    # A coordinate token is only populated for refinement cells that were
    # inserted between registered coarse coordinates.  Keeping it explicit in
    # the spec makes the lineage stable when a resumed campaign sees the same
    # adaptive boundary again.
    cell_token: str = ""

    @property
    def cell(self) -> tuple[float, float]:
        return (float(self.harm_strength), float(self.audit_sensitivity))

    def to_payload(self) -> dict[str, object]:
        return {
            "phase": str(self.phase),
            "level": int(self.level),
            "harm_index": int(self.harm_index),
            "audit_index": int(self.audit_index),
            "seed_index": int(self.seed_index),
            "harm_strength": float(self.harm_strength),
            "audit_sensitivity": float(self.audit_sensitivity),
            "model_seed": int(self.model_seed),
            "sampler_seed": int(self.sampler_seed),
            "run_id": str(self.run_id),
            "cell_token": str(self.cell_token),
        }


@dataclass(frozen=True, slots=True)
class BasinPhaseResult:
    """Compact result for one resumable bank phase."""

    specs: tuple[BasinReplicaSpec, ...]
    run_rows: tuple[dict[str, object], ...]
    checkpoint_rows: tuple[dict[str, object], ...]
    final_rows: tuple[dict[str, object], ...]
    pair_rows: tuple[dict[str, object], ...]
    evaluations: tuple[EvaluationMetrics, ...]
    bank_id: str


@dataclass(frozen=True, slots=True)
class BasinBankResult:
    """Result returned by :func:`run_basin_bank`."""

    tables: dict[str, list[dict[str, object]]]
    final_metrics: tuple[EvaluationMetrics, ...]
    coarse_cells: tuple[tuple[float, float], ...]
    refined_cells: tuple[tuple[int, float, float], ...]


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _grid(value: object, name: str, default: Sequence[float]) -> tuple[float, ...]:
    supplied = default if value is None else value
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of finite numbers")
    result = tuple(_finite_float(item, name) for item in supplied)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate coordinates")
    return result


def _positive_int(value: object, name: str, default: int) -> int:
    supplied = default if value is None else value
    if isinstance(supplied, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        parsed = int(supplied)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _coordinate_token(harm: float, audit: float) -> str:
    """Return a short, deterministic token for an arbitrary cell coordinate."""

    # ``float.hex`` preserves signed zero and subnormal values without locale
    # or formatting ambiguity.  The digest keeps remote run IDs compact while
    # remaining deterministic across Python versions.
    payload = f"{float(harm).hex()}|{float(audit).hex()}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _coordinate_indices(
    selected: Sequence[tuple[float, float]],
    harm_grid: Sequence[float],
    audit_grid: Sequence[float],
    *,
    allow_external: bool,
) -> tuple[tuple[int, int, float, float, str], ...]:
    """Resolve grid indices while permitting deterministic refinement points."""

    harm_positions = {float(value): index for index, value in enumerate(harm_grid)}
    audit_positions = {float(value): index for index, value in enumerate(audit_grid)}
    external_harm = sorted({float(harm) for harm, _ in selected if float(harm) not in harm_positions})
    external_audit = sorted({float(audit) for _, audit in selected if float(audit) not in audit_positions})
    if (external_harm or external_audit) and not allow_external:
        raise ValueError("coarse cells must be present in the registered grid")
    external_harm_positions = {
        value: len(harm_grid) + index for index, value in enumerate(external_harm)
    }
    external_audit_positions = {
        value: len(audit_grid) + index for index, value in enumerate(external_audit)
    }
    result: list[tuple[int, int, float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    for harm, audit in selected:
        harm_value = _finite_float(harm, "harm_strength")
        audit_value = _finite_float(audit, "audit_sensitivity")
        key = (harm_value, audit_value)
        if key in seen:
            raise ValueError("refinement cells must not contain duplicate coordinates")
        seen.add(key)
        harm_index = harm_positions.get(harm_value, external_harm_positions.get(harm_value))
        audit_index = audit_positions.get(audit_value, external_audit_positions.get(audit_value))
        if harm_index is None or audit_index is None:
            raise ValueError("cell coordinate could not be resolved")
        token = "" if harm_value in harm_positions and audit_value in audit_positions else _coordinate_token(
            harm_value, audit_value
        )
        result.append((int(harm_index), int(audit_index), harm_value, audit_value, token))
    return tuple(sorted(result, key=lambda item: (item[2], item[3], item[0], item[1])))


def _cell_indices(
    cells: Sequence[tuple[float, float]],
    harm_grid: Sequence[float],
    audit_grid: Sequence[float],
) -> tuple[tuple[int, int, float, float], ...]:
    return tuple(item[:4] for item in _coordinate_indices(cells, harm_grid, audit_grid, allow_external=False))


def build_basin_specs(
    values: Mapping[str, object],
    *,
    phase: str = "coarse",
    level: int = 0,
    cells: Sequence[tuple[float, float]] | None = None,
    seed_starts: Mapping[tuple[float, float], int] | None = None,
) -> tuple[BasinReplicaSpec, ...]:
    """Build deterministic coarse or refinement identities from config.

    ``seed_starts`` is used only for refinement phases.  It records the
    already-used seed count in each cell, so a resumed campaign never reuses a
    refinement run id.
    """

    phase_value = str(phase).strip().lower()
    if phase_value not in {"coarse", "refinement"}:
        raise ValueError("phase must be 'coarse' or 'refinement'")
    if isinstance(level, bool) or int(level) < 0:
        raise ValueError("level must be a non-negative integer")
    level = int(level)
    harm_grid = _grid(values.get("harmful_goal_grid"), "harmful_goal_grid", (2.0,))
    audit_grid = _grid(values.get("audit_sensitivity_grid"), "audit_sensitivity_grid", (0.0,))
    if cells is None:
        selected = tuple((harm, audit) for harm in harm_grid for audit in audit_grid)
    else:
        selected = tuple((float(pair[0]), float(pair[1])) for pair in cells)
    coordinates = _coordinate_indices(
        selected,
        harm_grid,
        audit_grid,
        allow_external=phase_value == "refinement",
    )
    coarse_count = _positive_int(values.get("seeds_per_cell"), "seeds_per_cell", 8)
    refinement_count = _positive_int(values.get("refinement_seeds"), "refinement_seeds", 16)
    count = coarse_count if phase_value == "coarse" else refinement_count
    starts = seed_starts or {}
    specs: list[BasinReplicaSpec] = []
    for harm_index, audit_index, harm, audit, cell_token in coordinates:
        start = 0 if phase_value == "coarse" else int(starts.get((harm, audit), 0))
        if start < 0:
            raise ValueError("refinement seed starts must be non-negative")
        for offset in range(count):
            seed_index = start + offset
            if phase_value == "coarse":
                model_seed = 500_000 + harm_index * 10_000 + audit_index * 100 + seed_index
                sampler_seed = 600_000 + harm_index * 10_000 + audit_index * 100 + seed_index
                run_id = f"basin-h{harm_index}-a{audit_index}-r{seed_index}"
            else:
                if cell_token:
                    # Hash-derived seeds avoid collisions with the legacy
                    # coarse/indexed formula when adaptive points are added.
                    digest = hashlib.sha256(
                        f"basin-refinement|{level}|{cell_token}".encode("ascii")
                    ).digest()
                    coordinate_offset = int.from_bytes(digest[:6], "big") % 80_000
                    model_seed = 1_000_000 + level * 100_000 + coordinate_offset + seed_index
                    sampler_seed = 1_100_000 + level * 100_000 + coordinate_offset + seed_index
                    run_id = f"basin-refine{level}-c{cell_token}-r{seed_index}"
                else:
                    model_seed = (
                        800_000
                        + level * 100_000
                        + harm_index * 10_000
                        + audit_index * 100
                        + seed_index
                    )
                    sampler_seed = (
                        900_000
                        + level * 100_000
                        + harm_index * 10_000
                        + audit_index * 100
                        + seed_index
                    )
                    run_id = f"basin-refine{level}-h{harm_index}-a{audit_index}-r{seed_index}"
            specs.append(
                BasinReplicaSpec(
                    phase=phase_value,
                    level=level,
                    harm_index=harm_index,
                    audit_index=audit_index,
                    seed_index=seed_index,
                    harm_strength=harm,
                    audit_sensitivity=audit,
                    model_seed=model_seed,
                    sampler_seed=sampler_seed,
                    run_id=run_id,
                    cell_token=cell_token,
                )
            )
    if not specs:
        raise ValueError("basin phase has no replicas")
    run_ids = [spec.run_id for spec in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("basin phase generated duplicate run ids")
    return tuple(specs)


def select_refinement_cells(
    finals_by_cell: Mapping[tuple[float, float], Sequence[Mapping[str, object]]],
    *,
    harmful_goal_grid: Sequence[float],
    audit_sensitivity_grid: Sequence[float],
    probability_low: float = 0.20,
    probability_high: float = 0.80,
    neighbor_delta: float = 0.40,
) -> tuple[tuple[float, float], ...]:
    """Select uncertain or sharply changing cells using all endpoint modes.

    The refinement decision is made from final rows only.  A cell is retained
    when any of the three registered labels has uncertain support, or when a
    neighboring cell changes materially for any label.  Using only the
    invariant probability can miss a strategic/intermediate boundary when the
    invariant mass is already near zero.
    """

    harm_grid = tuple(_finite_float(item, "harmful_goal_grid") for item in harmful_goal_grid)
    audit_grid = tuple(_finite_float(item, "audit_sensitivity_grid") for item in audit_sensitivity_grid)
    if not harm_grid or not audit_grid:
        raise ValueError("refinement grids must be non-empty")
    low = _finite_float(probability_low, "probability_low")
    high = _finite_float(probability_high, "probability_high")
    delta = _finite_float(neighbor_delta, "neighbor_delta")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("refinement probability thresholds must satisfy 0 <= low <= high <= 1")
    if delta < 0.0:
        raise ValueError("neighbor_delta must be non-negative")

    mode_names = tuple(mode.value for mode in Mode)
    probabilities: dict[tuple[float, float], dict[str, float]] = {}
    observed_cells: set[tuple[float, float]] = set()
    for harm in harm_grid:
        for audit in audit_grid:
            key = (harm, audit)
            rows = tuple(finals_by_cell.get(key, ()))
            if not rows:
                probabilities[key] = {name: 0.0 for name in mode_names}
                continue
            observed_cells.add(key)
            counts = {name: 0 for name in mode_names}
            for row in rows:
                label = str(row.get("label", row.get("final_label", Mode.INTERMEDIATE.value)))
                if label not in counts:
                    label = Mode.INTERMEDIATE.value
                counts[label] += 1
            probabilities[key] = {
                name: count / len(rows) for name, count in counts.items()
            }

    selected: set[tuple[float, float]] = set()
    for harm_index, harm in enumerate(harm_grid):
        for audit_index, audit in enumerate(audit_grid):
            key = (harm, audit)
            if key not in observed_cells:
                continue
            cell_probabilities = probabilities[key]
            uncertain = any(low <= probability <= high for probability in cell_probabilities.values())
            if uncertain:
                selected.add(key)
                continue
            neighbors: list[tuple[float, float]] = []
            if harm_index:
                neighbors.append((harm_grid[harm_index - 1], audit))
            if harm_index + 1 < len(harm_grid):
                neighbors.append((harm_grid[harm_index + 1], audit))
            if audit_index:
                neighbors.append((harm, audit_grid[audit_index - 1]))
            if audit_index + 1 < len(audit_grid):
                neighbors.append((harm, audit_grid[audit_index + 1]))
            if any(
                neighbor in probabilities
                and neighbor in observed_cells
                and any(
                    abs(cell_probabilities[name] - probabilities[neighbor][name]) >= delta
                    for name in mode_names
                )
                for neighbor in neighbors
            ):
                selected.add(key)
    return tuple(sorted(selected))


def densify_boundary_cells(
    finals_by_cell: Mapping[tuple[float, float], Sequence[Mapping[str, object]]],
    selected_cells: Sequence[tuple[float, float]],
    *,
    harmful_goal_grid: Sequence[float],
    audit_sensitivity_grid: Sequence[float],
    max_new_cells: int | None = None,
) -> tuple[tuple[float, float], ...]:
    """Create new midpoint coordinates around selected basin boundaries.

    ``selected_cells`` are the cells flagged by :func:`select_refinement_cells`.
    The axis coordinates include previously inserted points, so later levels
    produce quarter- or finer-scale samples rather than repeating the coarse
    grid.  Existing evaluated coordinates are omitted.  The helper remains
    deterministic under arbitrary mapping and input sequence order.
    """

    harm_grid = tuple(sorted(set(_finite_float(item, "harmful_goal_grid") for item in harmful_goal_grid)))
    audit_grid = tuple(sorted(set(_finite_float(item, "audit_sensitivity_grid") for item in audit_sensitivity_grid)))
    if not harm_grid or not audit_grid:
        raise ValueError("refinement grids must be non-empty")
    selected = tuple(
        (_finite_float(pair[0], "harm_strength"), _finite_float(pair[1], "audit_sensitivity"))
        for pair in selected_cells
    )
    if len(set(selected)) != len(selected):
        raise ValueError("selected refinement cells must not contain duplicates")
    if max_new_cells is not None:
        if isinstance(max_new_cells, bool) or int(max_new_cells) < 1:
            raise ValueError("max_new_cells must be positive when supplied")
        max_new_cells = int(max_new_cells)

    known = {
        (_finite_float(pair[0], "harm_strength"), _finite_float(pair[1], "audit_sensitivity"))
        for pair in finals_by_cell
    }
    harm_axis = tuple(sorted(set(harm_grid) | {cell[0] for cell in known} | {cell[0] for cell in selected}))
    audit_axis = tuple(sorted(set(audit_grid) | {cell[1] for cell in known} | {cell[1] for cell in selected}))
    new_cells: set[tuple[float, float]] = set()

    def _midpoints(axis: Sequence[float], coordinate: float) -> tuple[float, ...]:
        try:
            index = axis.index(coordinate)
        except ValueError:
            return ()
        values: list[float] = []
        if index:
            values.append((float(axis[index - 1]) + coordinate) / 2.0)
        if index + 1 < len(axis):
            values.append((coordinate + float(axis[index + 1])) / 2.0)
        return tuple(value for value in values if math.isfinite(value) and value != coordinate)

    for harm, audit in sorted(set(selected)):
        for midpoint in _midpoints(harm_axis, harm):
            candidate = (midpoint, audit)
            if candidate not in known:
                new_cells.add(candidate)
        for midpoint in _midpoints(audit_axis, audit):
            candidate = (harm, midpoint)
            if candidate not in known:
                new_cells.add(candidate)
    ordered = tuple(sorted(new_cells))
    if max_new_cells is not None:
        return ordered[:max_new_cells]
    return ordered


def _annotate_row(
    row: Mapping[str, object],
    spec: BasinReplicaSpec,
    *,
    experiment: str = "toy_basin",
) -> dict[str, object]:
    value = dict(row)
    value.update(
        {
            "experiment": experiment,
            "architecture": "toy",
            "condition": "initial_condition_basin",
            "phase": spec.phase,
            "level": spec.level,
            "harm_strength": spec.harm_strength,
            "audit_sensitivity": spec.audit_sensitivity,
            "harm_index": spec.harm_index,
            "audit_index": spec.audit_index,
            "seed_index": spec.seed_index,
            "cell_token": spec.cell_token,
        }
    )
    return value


def _phase_manifest(
    *,
    bank_id: str,
    specs: Sequence[BasinReplicaSpec],
    target_steps: int,
    config_identity: str,
    source_identity: str,
    dataset_hash: str,
    eval_hash: str,
    bank: ReplicaBank,
) -> dict[str, object]:
    first = specs[0]
    return {
        "schema_version": BASIN_BANK_SCHEMA_VERSION,
        "bank_id": bank_id,
        "experiment": "toy_basin",
        "architecture": "toy",
        "phase": first.phase,
        "level": first.level,
        "config_sha256": config_identity,
        "source_archive_sha256": source_identity,
        "dataset_sha256": dataset_hash,
        "evaluation_sha256": eval_hash,
        "target_steps": int(target_steps),
        "specs": [spec.to_payload() for spec in specs],
        "engine": bank.manifest(),
    }


def _phase_checkpoint_config(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "basin_bank_schema_version": BASIN_BANK_SCHEMA_VERSION,
        "bank_id": manifest["bank_id"],
        "phase": manifest["phase"],
        "level": manifest["level"],
        "config_sha256": manifest["config_sha256"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "evaluation_sha256": manifest["evaluation_sha256"],
        "target_steps": manifest["target_steps"],
        "specs": manifest["specs"],
    }


def _run_phase_bank(
    *,
    session_dir: Path,
    values: Mapping[str, object],
    specs: Sequence[BasinReplicaSpec],
    config_identity: str,
    source_identity: str,
    target_steps: int,
    train_episodes: Sequence[object],
    eval_pairs: Sequence[PairedEpisode],
    dataset_hash: str,
    eval_hash: str,
    thresholds: ModeThresholds,
) -> BasinPhaseResult:
    """Train one coarse/refinement bank and recover it after interruption."""

    from . import campaign as core
    from .bank_campaign import _checkpoint_metrics, _evaluate_bank, _loaded_user_metadata, _validate_raw_rows

    replicas = len(specs)
    if replicas < 1:
        raise ValueError("basin bank must contain at least one replica")
    device = str(values.get("device", "cuda"))
    models = [
        core._model(
            "toy",
            task_dim=int(values.get("task_dim", 4)),
            hidden_width=int(values.get("hidden_width", 16)),
            harmful_goal_strength=float(spec.harm_strength),
            initial_audit_sensitivity=float(spec.audit_sensitivity),
            model_seed=int(spec.model_seed),
            device="cpu",
        )
        for spec in specs
    ]
    bank = ReplicaBank.from_agents(
        models,
        model_seeds=tuple(spec.model_seed for spec in specs),
        sampler_seeds=tuple(spec.sampler_seed for spec in specs),
        device=device,
        learning_rate=float(values.get("learning_rate", 0.003)),
    )
    first = specs[0]
    bank_id = f"toy-basin-{first.phase}-l{first.level}-bank"
    store = CheckpointStore(
        session_dir / "banks",
        bank_id,
        config_identity=config_identity,
        source_identity=source_identity,
    )
    manifest = _phase_manifest(
        bank_id=bank_id,
        specs=specs,
        target_steps=target_steps,
        config_identity=config_identity,
        source_identity=source_identity,
        dataset_hash=dataset_hash,
        eval_hash=eval_hash,
        bank=bank,
    )
    manifest_path = store.run_dir / "manifest.json"
    if manifest_path.is_file():
        existing = core._read_json(manifest_path)
        for key in (
            "schema_version",
            "bank_id",
            "phase",
            "level",
            "config_sha256",
            "source_archive_sha256",
            "dataset_sha256",
            "evaluation_sha256",
            "target_steps",
            "specs",
        ):
            if existing.get(key) != manifest.get(key):
                raise core.CampaignError(f"basin bank manifest mismatch for {key}")
        if json.dumps(existing.get("engine"), sort_keys=True, separators=(",", ":")) != json.dumps(
            manifest.get("engine"), sort_keys=True, separators=(",", ":")
        ):
            raise core.CampaignError("basin bank manifest engine identity mismatch")
    else:
        core._write_json(manifest_path, manifest)

    checkpoint_config = _phase_checkpoint_config(manifest)
    metrics_path = session_dir / "raw" / "basin_checkpoint_metrics.jsonl"
    existing_rows = core._read_jsonl(metrics_path)
    run_ids = tuple(spec.run_id for spec in specs)
    run_id_set = set(run_ids)
    expected_pairs = sum(bool(pair.on.secret_opportunity) for pair in eval_pairs)
    try:
        logged = _validate_raw_rows(
            existing_rows,
            run_ids=run_ids,
            eval_hash=eval_hash,
            target_steps=target_steps,
            expected_pairs=expected_pairs,
        )
    except ValueError as exc:
        raise core.CampaignError(str(exc)) from exc
    resume_state: BatchedTrainingState | None = None
    latest_metrics: Mapping[str, object] | None = None
    latest = store.latest()
    if latest is not None:
        if int(latest.step) > int(target_steps):
            raise core.CampaignError("basin bank checkpoint is beyond its registered target step")
        loaded = store.load(latest, load_torch=True)
        if not isinstance(loaded.torch_state, Mapping):
            raise core.CampaignError("basin bank checkpoint has no torch payload")
        resume_state, metadata = bank.load_checkpoint_payload(
            loaded.torch_state,
            map_location=device,
            expected_config=checkpoint_config,
        )
        user_metadata = _loaded_user_metadata(loaded)
        if (
            user_metadata.get("dataset_sha256") != dataset_hash
            or user_metadata.get("evaluation_sha256") != eval_hash
            or user_metadata.get("source_archive_sha256") != source_identity
            or user_metadata.get("bank_id") != bank_id
        ):
            raise core.CampaignError("basin bank checkpoint provenance mismatch")
        candidate = user_metadata.get("metrics")
        latest_metrics = candidate if isinstance(candidate, Mapping) else None

    def evaluate_and_log(
        state: BatchedTrainingState,
        metrics: Mapping[str, object] | None,
    ) -> None:
        _evaluated, rows, pair_rows = _evaluate_bank(
            bank,
            eval_pairs,
            run_ids=run_ids,
            eval_hash=eval_hash,
            thresholds=thresholds,
            step=int(state.global_step),
            target_steps=target_steps,
            training_metrics=metrics,
        )
        for index, (row, pair_row) in enumerate(zip(rows, pair_rows, strict=True)):
            annotated = _annotate_row(row, specs[index])
            key = (str(annotated["run_id"]), int(annotated["step"]))
            if key not in logged:
                core._append_jsonl(metrics_path, annotated)
                logged.add(key)
            # Pair rows are not stored in the raw trajectory.  The final
            # compact table is rebuilt from the current bank state below.

    if resume_state is not None and any(
        (run_id, int(resume_state.global_step)) not in logged for run_id in run_ids
    ):
        evaluate_and_log(resume_state, latest_metrics)

    def checkpoint_callback(
        *,
        bank: ReplicaBank,
        state: BatchedTrainingState,
        metrics: Mapping[str, object],
        **_: object,
    ) -> None:
        nonlocal latest_metrics
        metrics_payload = _checkpoint_metrics(metrics, replicas)
        latest_metrics = metrics_payload
        payload = bank.checkpoint_payload(
            state=state,
            config=checkpoint_config,
            metadata={
                "phase": first.phase,
                "level": first.level,
                "dataset_sha256": dataset_hash,
                "evaluation_sha256": eval_hash,
                "source_archive_sha256": source_identity,
                "metrics": metrics_payload,
            },
        )
        store.save(
            int(state.global_step),
            {
                "global_step": int(state.global_step),
                "epoch": int(state.epoch),
                "batch_offset": int(state.batch_offset),
                "replica_count": replicas,
            },
            optimizer_state={"storage": "torch_state.pt"},
            rng_state={"sampler_seeds": [spec.sampler_seed for spec in specs]},
            minibatch_cursor=int(state.batch_offset),
            torch_state=payload,
            metadata={
                "phase": first.phase,
                "level": first.level,
                "bank_id": bank_id,
                "dataset_sha256": dataset_hash,
                "evaluation_sha256": eval_hash,
                "source_archive_sha256": source_identity,
                "metrics": metrics_payload,
            },
        )
        evaluate_and_log(state, metrics_payload)

    training_config = core._training_config(
        values,
        target_steps=target_steps,
        model_seed=int(first.model_seed),
        sampler_seed=int(first.sampler_seed),
    )
    completed = store.is_run_complete() and latest is not None and int(latest.step) == int(target_steps)
    if completed:
        try:
            completion = core._read_json(store.run_dir / "RUN_COMPLETE.json")
            summary = completion.get("summary")
            completed = (
                isinstance(summary, Mapping)
                and int(summary.get("final_step", -1)) == int(target_steps)
                and summary.get("dataset_sha256") == dataset_hash
                and summary.get("evaluation_sha256", eval_hash) == eval_hash
                and summary.get("source_archive_sha256", source_identity) == source_identity
            )
        except (core.CampaignError, OSError, ValueError, TypeError, json.JSONDecodeError):
            completed = False
    if not completed:
        final_state = bank.train(
            train_episodes,
            training_config,
            RewardConfig(),
            state=resume_state,
            checkpoint_callback=checkpoint_callback,
        )
        if int(final_state.global_step) != int(target_steps):
            raise core.CampaignError("basin bank training stopped before its registered target")
        store.mark_run_complete(
            summary={
                "final_step": int(final_state.global_step),
                "dataset_sha256": dataset_hash,
                "evaluation_sha256": eval_hash,
                "source_archive_sha256": source_identity,
                "replica_count": replicas,
                "phase": first.phase,
                "level": first.level,
            }
        )

    latest = store.latest()
    if latest is None:
        raise core.CampaignError("completed basin bank has no checkpoint")
    if latest.step != target_steps:
        raise core.CampaignError("basin bank completed without its exact target checkpoint")
    final_train_metrics = latest_metrics
    if final_train_metrics is None:
        final_train_metrics = {
            "expected_reward": [None] * replicas,
            "loss": [None] * replicas,
        }
    evaluated, refreshed_rows, pair_rows = _evaluate_bank(
        bank,
        eval_pairs,
        run_ids=run_ids,
        eval_hash=eval_hash,
        thresholds=thresholds,
        step=target_steps,
        target_steps=target_steps,
        training_metrics=final_train_metrics,
    )
    final_rows = tuple(
        _annotate_row({**row, "is_final": True}, specs[index])
        for index, row in enumerate(refreshed_rows)
    )
    final_pair_rows = tuple(_annotate_row(row, specs[index]) for index, row in enumerate(pair_rows))
    all_raw = core._read_jsonl(metrics_path)
    try:
        _validate_raw_rows(
            all_raw,
            run_ids=run_ids,
            eval_hash=eval_hash,
            target_steps=target_steps,
            expected_pairs=expected_pairs,
        )
    except ValueError as exc:
        raise core.CampaignError(str(exc)) from exc
    raw_final_ids = {
        str(row.get("run_id"))
        for row in all_raw
        if str(row.get("step")) == str(target_steps)
    }
    if raw_final_ids != run_id_set:
        if resume_state is not None:
            evaluate_and_log(resume_state, latest_metrics)
            all_raw = core._read_jsonl(metrics_path)
            try:
                _validate_raw_rows(
                    all_raw,
                    run_ids=run_ids,
                    eval_hash=eval_hash,
                    target_steps=target_steps,
                    expected_pairs=expected_pairs,
                )
            except ValueError as exc:
                raise core.CampaignError(str(exc)) from exc
            raw_final_ids = {
                str(row.get("run_id"))
                for row in all_raw
                if str(row.get("step")) == str(target_steps)
            }
        if raw_final_ids != run_id_set:
            raise core.CampaignError("basin bank is missing exact final trajectory rows")
    phase_rows = tuple(row for row in all_raw if str(row.get("run_id")) in run_id_set)
    run_rows: list[dict[str, object]] = []
    for index, (spec, row, metric) in enumerate(zip(specs, final_rows, evaluated, strict=True)):
        run_rows.append(
            {
                "schema_version": core.SCHEMA_VERSION,
                "experiment": "toy_basin",
                "architecture": "toy",
                "condition": "initial_condition_basin",
                "phase": spec.phase,
                "level": spec.level,
                "harm_strength": spec.harm_strength,
                "audit_sensitivity": spec.audit_sensitivity,
                "harm_index": spec.harm_index,
                "audit_index": spec.audit_index,
                "seed_index": spec.seed_index,
                "cell_token": spec.cell_token,
                "run_id": spec.run_id,
                "seed": spec.model_seed,
                "shuffle_seed": spec.sampler_seed,
                "config_sha256": config_identity,
                "git_commit": core.os.environ.get("RH_SOURCE_COMMIT", "unknown"),
                "status": "complete",
                "final_step": target_steps,
                "final_label": metric.mode.value if metric.mode else Mode.INTERMEDIATE.value,
                "threshold_set": "readme_primary",
                "remote_raw_uri": f"{store.run_dir}#replica={index}",
                "checkpoint_count": len(list(store.run_dir.glob("checkpoint-*"))),
            }
        )
    return BasinPhaseResult(
        specs=tuple(specs),
        run_rows=tuple(run_rows),
        checkpoint_rows=phase_rows,
        final_rows=final_rows,
        pair_rows=final_pair_rows,
        evaluations=tuple(evaluated),
        bank_id=bank_id,
    )


def _basin_cells_table(
    finals_by_cell: Mapping[tuple[float, float], Sequence[Mapping[str, object]]],
    *,
    config_identity: str,
    alpha: float,
) -> list[dict[str, object]]:
    from . import campaign as core

    rows: list[dict[str, object]] = []
    for (harm, audit), finals in sorted(finals_by_cell.items()):
        labels = [str(row.get("label", Mode.INTERMEDIATE.value)) for row in finals]
        count = len(labels)
        invariant = labels.count(Mode.OVERSIGHT_INVARIANT.value)
        strategic = labels.count(Mode.STRATEGIC.value)
        intermediate = labels.count(Mode.INTERMEDIATE.value)
        ci_low, ci_high = core.wilson_interval(invariant, count, alpha=float(alpha))
        rows.append(
            {
                "schema_version": core.SCHEMA_VERSION,
                "experiment": "toy_basin",
                "architecture": "toy",
                "harm_strength": float(harm),
                "audit_sensitivity": float(audit),
                "n_seeds": count,
                "n_complete": count,
                "n_invariant": invariant,
                "n_strategic": strategic,
                "n_intermediate": intermediate,
                "p_invariant": invariant / count if count else 0.0,
                "p_strategic": strategic / count if count else 0.0,
                "p_intermediate": intermediate / count if count else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "config_sha256": config_identity,
            }
        )
    return rows


def run_basin_bank(
    *,
    session_dir: Path,
    values: Mapping[str, object],
    config_identity: str,
    source_identity: str,
    target_steps: int,
    train_episodes: Sequence[object],
    eval_pairs: Sequence[PairedEpisode],
    dataset_hash: str | None = None,
    eval_hash: str | None = None,
    thresholds: ModeThresholds | None = None,
) -> BasinBankResult:
    """Run the coarse basin map and registered adaptive refinements.

    Every phase is a separate bank checkpoint lineage.  A process killed after
    any committed checkpoint resumes from that phase's ``latest.json``; raw
    rows and manifests remain append-only and hash-bound to the config,
    source, training data, and paired evaluation set.
    """

    from . import campaign as core
    from .safety import assert_colab_execution

    if isinstance(target_steps, bool) or int(target_steps) < 1:
        raise ValueError("target_steps must be a positive integer")
    target_steps = int(target_steps)
    session_dir = Path(session_dir)
    # Guard the public API before model construction.  In particular, a
    # direct caller must not be able to allocate a CUDA bank and encounter the
    # Colab check only after the allocation has happened.
    requested_device = str(values.get("device", "cuda")).strip().lower()
    assert_colab_execution(
        require_gpu=requested_device.startswith(("cuda", "mps", "gpu"))
    )
    computed_train_hash = dataset_fingerprint(train_episodes)
    computed_eval_hash = dataset_fingerprint(eval_pairs)
    if dataset_hash is not None and str(dataset_hash) != computed_train_hash:
        raise core.CampaignError("basin training dataset hash does not match episode content")
    if eval_hash is not None and str(eval_hash) != computed_eval_hash:
        raise core.CampaignError("basin evaluation hash does not match paired episode content")
    train_hash = computed_train_hash
    eval_fingerprint = computed_eval_hash
    threshold_values = thresholds or ModeThresholds()
    harm_grid = _grid(values.get("harmful_goal_grid"), "harmful_goal_grid", (2.0,))
    audit_grid = _grid(values.get("audit_sensitivity_grid"), "audit_sensitivity_grid", (0.0,))
    coarse_specs = build_basin_specs(values)
    coarse_phase = _run_phase_bank(
        session_dir=session_dir,
        values=values,
        specs=coarse_specs,
        config_identity=config_identity,
        source_identity=source_identity,
        target_steps=target_steps,
        train_episodes=train_episodes,
        eval_pairs=eval_pairs,
        dataset_hash=train_hash,
        eval_hash=eval_fingerprint,
        thresholds=threshold_values,
    )
    all_phases = [coarse_phase]
    finals_by_cell: dict[tuple[float, float], list[dict[str, object]]] = {}
    for spec, row in zip(coarse_specs, coarse_phase.final_rows, strict=True):
        finals_by_cell.setdefault(spec.cell, []).append(row)

    low = _finite_float(values.get("refinement_probability_low", 0.20), "refinement_probability_low")
    high = _finite_float(values.get("refinement_probability_high", 0.80), "refinement_probability_high")
    neighbor_delta = _finite_float(values.get("refinement_neighbor_delta", 0.40), "refinement_neighbor_delta")
    max_levels = int(values.get("max_refinement_levels", 0))
    if max_levels < 0:
        raise ValueError("max_refinement_levels must be non-negative")
    refined_cells: list[tuple[int, float, float]] = []
    seen_refined_cells: set[tuple[float, float]] = set()
    coarse_cells = tuple((float(harm), float(audit)) for harm in harm_grid for audit in audit_grid)
    for level in range(1, max_levels + 1):
        # Previous refinement coordinates join the axes before boundary
        # selection.  This lets the midpoint helper place genuinely new
        # points at successively finer scales.
        candidate_harm_grid = tuple(sorted({float(harm) for harm, _ in finals_by_cell} | set(harm_grid)))
        candidate_audit_grid = tuple(sorted({float(audit) for _, audit in finals_by_cell} | set(audit_grid)))
        cells = select_refinement_cells(
            finals_by_cell,
            harmful_goal_grid=candidate_harm_grid,
            audit_sensitivity_grid=candidate_audit_grid,
            probability_low=low,
            probability_high=high,
            neighbor_delta=neighbor_delta,
        )
        cells = densify_boundary_cells(
            finals_by_cell,
            cells,
            harmful_goal_grid=candidate_harm_grid,
            audit_sensitivity_grid=candidate_audit_grid,
        )
        # A resumed run can encounter a boundary that was already completed
        # by a previous process.  Only unseen coordinates create a new phase.
        cells = tuple(cell for cell in cells if cell not in finals_by_cell)
        if not cells:
            break
        starts = {cell: len(finals_by_cell.get(cell, ())) for cell in cells}
        refinement_specs = build_basin_specs(
            values,
            phase="refinement",
            level=level,
            cells=cells,
            seed_starts=starts,
        )
        phase = _run_phase_bank(
            session_dir=session_dir,
            values=values,
            specs=refinement_specs,
            config_identity=config_identity,
            source_identity=source_identity,
            target_steps=target_steps,
            train_episodes=train_episodes,
            eval_pairs=eval_pairs,
            dataset_hash=train_hash,
            eval_hash=eval_fingerprint,
            thresholds=threshold_values,
        )
        all_phases.append(phase)
        for spec, row in zip(refinement_specs, phase.final_rows, strict=True):
            finals_by_cell.setdefault(spec.cell, []).append(row)
            if spec.cell not in seen_refined_cells:
                seen_refined_cells.add(spec.cell)
                refined_cells.append((level, spec.harm_strength, spec.audit_sensitivity))

    table_names = getattr(core, "TABLE_NAMES", (
        "runs.csv",
        "pair_counts.csv",
        "checkpoint_metrics.csv",
        "final_summary.csv",
        "basin_cells.csv",
    ))
    tables: dict[str, list[dict[str, object]]] = {str(name): [] for name in table_names}
    for phase in all_phases:
        tables.setdefault("runs.csv", []).extend(dict(row) for row in phase.run_rows)
        tables.setdefault("checkpoint_metrics.csv", []).extend(dict(row) for row in phase.checkpoint_rows)
        tables.setdefault("final_summary.csv", []).extend(dict(row) for row in phase.final_rows)
        tables.setdefault("pair_counts.csv", []).extend(dict(row) for row in phase.pair_rows)
    statistics = values.get("statistics")
    if isinstance(statistics, Mapping):
        alpha_value = statistics.get("alpha", values.get("alpha", 0.05))
    else:
        alpha_value = getattr(statistics, "alpha", values.get("alpha", 0.05))
    alpha = _finite_float(alpha_value, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    tables.setdefault("basin_cells.csv", []).extend(
        _basin_cells_table(finals_by_cell, config_identity=config_identity, alpha=alpha)
    )
    summary = {
        "schema_version": BASIN_BANK_SCHEMA_VERSION,
        "experiment": "toy_basin",
        "config_sha256": config_identity,
        "source_archive_sha256": source_identity,
        "dataset_sha256": train_hash,
        "evaluation_sha256": eval_fingerprint,
        "target_steps": target_steps,
        "harmful_goal_grid": list(harm_grid),
        "audit_sensitivity_grid": list(audit_grid),
        "coarse_cells": [list(cell) for cell in sorted(coarse_cells)],
        "refined_cells": [list(item) for item in refined_cells],
        "phase_banks": [phase.bank_id for phase in all_phases],
    }
    core._write_json(session_dir / "raw" / "basin_manifest.json", summary)
    final_metrics = tuple(metric for phase in all_phases for metric in phase.evaluations)
    return BasinBankResult(
        tables=tables,
        final_metrics=final_metrics,
        coarse_cells=tuple(sorted(coarse_cells)),
        refined_cells=tuple(refined_cells),
    )


run_basin_sweep = run_basin_bank
run_basin_campaign = run_basin_bank


__all__ = [
    "BASIN_BANK_SCHEMA_VERSION",
    "BasinBankResult",
    "BasinPhaseResult",
    "BasinReplicaSpec",
    "build_basin_specs",
    "densify_boundary_cells",
    "run_basin_bank",
    "run_basin_campaign",
    "run_basin_sweep",
    "select_refinement_cells",
]
