#!/usr/bin/env python3
"""Build a data-bound local release from imported Colab compact bundles.

The release boundary is deliberately small.  It accepts only strict compact
bundles, checks that the four live experiment bundles are present, regenerates the
registered figures and derived tables, and writes a report whose claim ladder
is computed from those saved rows.  The language-model package is optional:
its package-ready status is recorded without supplying empirical LM evidence.

Raw checkpoints, model weights, logs, and remote run directories never enter
the local release tree.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


PIPELINE_VERSION = "2026-08-27.1"
REQUIRED_LIVE_BUNDLES = {
    "toy": ("toy_fixed", ("runs.csv", "checkpoint_metrics.csv")),
    "generic": (
        "generic_mlp",
        ("runs.csv", "checkpoint_metrics.csv", "audit_control.csv"),
    ),
    "basin": ("toy_basin", ("basin_cells.csv",)),
    "perturbation": ("toy_perturbation", ("perturbation_trajectory.csv",)),
}
LM_EXPERIMENT = "red_token_lm"
IDENTITY_FIELDS = (
    "config_sha256",
    "source_archive_sha256",
    "train_dataset_sha256",
    "eval_dataset_sha256",
    "reward_sha256",
    "objective_sha256",
)
RUN_PROVENANCE_FIELDS = ("seed", "shuffle_seed", "git_commit", *IDENTITY_FIELDS)
FIGURE_NAMES = (
    "fig01_final_gap_histogram.svg",
    "fig02_training_trajectories.svg",
    "fig03_basin_phase_diagram.svg",
    "fig04_perturbation_recovery.svg",
    "fig05_reward_vs_hidden_misalignment.svg",
    "fig06_control_audit_swaps.svg",
)
FIGURE_SOURCE_KEYS = {
    "final_gap": "toy",
    "trajectories": "toy",
    "basin": "basin",
    "perturbation": "perturbation",
    "reward": "toy",
    "control": "generic",
}
TABLE_NAMES = (
    "runs.csv",
    "pair_counts.csv",
    "checkpoint_metrics.csv",
    "final_summary.csv",
    "basin_cells.csv",
    "perturbation_trajectory.csv",
    "audit_control.csv",
    "threshold_sensitivity.csv",
)


class ReleaseError(RuntimeError):
    """Raised when an imported release cannot be verified fail-closed."""


@dataclass(frozen=True)
class BundleSpec:
    """One required or optional experiment input."""

    key: str
    experiment: str
    required_tables: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedBundle:
    """Metadata for a strict, imported compact bundle."""

    spec: BundleSpec
    path: Path
    manifest: dict[str, object]
    provenance: dict[str, object]
    manifest_path: Path
    manifest_sha256: str
    source_archive_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseError(f"JSON object required: {path}")
    return payload


def _manifest_file(root: Path) -> Path:
    candidates = [root / name for name in ("bundle_manifest.json", "manifest.json") if (root / name).is_file()]
    if len(candidates) != 1:
        raise ReleaseError(f"expected exactly one compact manifest in {root}")
    return candidates[0]


def _experiment_from_manifest(manifest: Mapping[str, object]) -> str:
    for field in ("analysis_experiment", "experiment_scope"):
        value = manifest.get(field)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _json_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReleaseError(f"missing provenance file: {path}")
    return _read_json_object(path)


def verify_imported_bundle(
    path: str | Path,
    *,
    key: str,
    experiment: str,
    required_tables: Sequence[str],
    optional: bool = False,
) -> VerifiedBundle:
    """Validate one imported bundle and require live rows for its experiment."""

    from lean_reward_hacking.schemas import (
        load_table,
        manifest_payload,
        table_path,
        validate_compact_bundle,
    )

    root = Path(path)
    problems = validate_compact_bundle(root, strict=True)
    if problems:
        prefix = "optional LM bundle" if optional else "compact bundle"
        raise ReleaseError(f"invalid {prefix} {root}: " + "; ".join(problems))
    manifest_path = _manifest_file(root)
    manifest = manifest_payload(root)
    actual_experiment = _experiment_from_manifest(manifest)
    if actual_experiment != experiment:
        raise ReleaseError(
            f"{key} bundle {root} names experiment {actual_experiment or 'missing'!r}; "
            f"expected {experiment!r}"
        )
    for table_name in required_tables:
        table = table_path(root, table_name)
        if table is None:
            raise ReleaseError(f"{key} bundle {root} is missing live table {table_name}")
        if not load_table(root, table_name):
            raise ReleaseError(f"{key} bundle {root} has no rows in live table {table_name}")

    runs = load_table(root, "runs.csv")
    if key in {"toy", "generic", "lm"} and not runs:
        raise ReleaseError(f"{key} bundle {root} has no run records")
    complete_runs = [
        row for row in runs
        if str(row.get("status") or "").strip().lower() in {"complete", "completed", "ok"}
    ]
    complete_ids = [str(row.get("run_id") or "").strip() for row in complete_runs]
    if len(complete_ids) != len(set(complete_ids)):
        raise ReleaseError(f"{key} bundle {root} repeats a complete run id")
    if key in {"toy", "generic", "lm"} and not complete_runs:
        raise ReleaseError(f"{key} bundle {root} has no complete live run")
    source_digest = str(manifest.get("source_archive_sha256") or "")
    if not _is_sha256(source_digest):
        raise ReleaseError(f"{key} bundle {root} has no valid source archive digest")
    provenance = _json_mapping(root / "provenance.json")
    provenance_record = manifest.get("provenance")
    if not isinstance(provenance_record, Mapping):
        raise ReleaseError(f"{key} bundle {root} manifest has no provenance record")
    if (
        provenance_record.get("path") != "provenance.json"
        or provenance_record.get("sha256") != _sha256_file(root / "provenance.json")
    ):
        raise ReleaseError(f"{key} bundle {root} provenance digest is unbound")
    if str(provenance.get("source_archive_sha256") or "") != source_digest:
        raise ReleaseError(f"{key} bundle {root} provenance source identity differs from its manifest")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReleaseError(f"{key} bundle {root} has no runtime provenance")
    packages = runtime.get("packages")
    accelerator = runtime.get("accelerator")
    if not isinstance(packages, Mapping) or not packages:
        raise ReleaseError(f"{key} bundle {root} has no dependency-version provenance")
    if not isinstance(accelerator, Mapping):
        raise ReleaseError(f"{key} bundle {root} has no accelerator provenance")
    accelerator_available = accelerator.get(
        "available", accelerator.get("torch_cuda_available")
    )
    accelerator_name = accelerator.get("name") or accelerator.get("torch_cuda_device")
    if accelerator_available is not True or not str(accelerator_name or "").strip():
        raise ReleaseError(f"{key} bundle {root} was not exported from a recorded GPU runtime")
    for row in complete_runs:
        missing_fields = [
            field for field in RUN_PROVENANCE_FIELDS
            if not str(row.get(field) or "").strip()
        ]
        invalid_hashes = [
            field for field in IDENTITY_FIELDS
            if field not in missing_fields and not _is_sha256(row.get(field))
        ]
        if missing_fields or invalid_hashes:
            raise ReleaseError(
                f"{key} bundle {root} complete run {row.get('run_id')!r} has invalid provenance: "
                + json.dumps(
                    {"missing": missing_fields, "invalid_hashes": invalid_hashes},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        if str(row.get("git_commit") or "").strip().lower() == "unknown":
            raise ReleaseError(
                f"{key} bundle {root} complete run {row.get('run_id')!r} has no Git commit"
            )
        if str(row.get("source_archive_sha256")) != source_digest:
            raise ReleaseError(
                f"{key} bundle {root} complete run {row.get('run_id')!r} has a different source archive"
            )
    return VerifiedBundle(
        spec=BundleSpec(key, experiment, tuple(str(item) for item in required_tables)),
        path=root,
        manifest=manifest,
        provenance=provenance,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        source_archive_sha256=source_digest,
    )


def _candidate_bundle(root: Path, spec: BundleSpec) -> Path | None:
    for candidate in (root / spec.experiment, root / spec.key):
        if candidate.is_dir():
            return candidate
    return None


def resolve_required_bundles(
    compact_root: str | Path,
    explicit: Mapping[str, str | Path | None] | None = None,
) -> dict[str, VerifiedBundle]:
    """Resolve and verify all live inputs before creating release outputs."""

    root = Path(compact_root)
    supplied = explicit or {}
    records: dict[str, VerifiedBundle] = {}
    missing: list[str] = []
    for key, (experiment, required_tables) in REQUIRED_LIVE_BUNDLES.items():
        spec = BundleSpec(key, experiment, tuple(required_tables))
        supplied_path = supplied.get(key)
        candidate = Path(supplied_path) if supplied_path is not None else _candidate_bundle(root, spec)
        if candidate is None:
            missing.append(f"{key} ({experiment})")
            continue
        records[key] = verify_imported_bundle(
            candidate,
            key=key,
            experiment=experiment,
            required_tables=required_tables,
        )
    if missing:
        raise ReleaseError(
            "missing live compact bundle(s): " + ", ".join(missing)
            + ". Supply --toy-bundle, --generic-bundle, --basin-bundle, and --perturbation-bundle "
            + "after importing completed Colab exports."
        )
    return records


def _optional_lm_bundle(path: str | Path | None) -> VerifiedBundle | None:
    if path is None:
        return None
    return verify_imported_bundle(
        path,
        key="lm",
        experiment=LM_EXPERIMENT,
        required_tables=("runs.csv", "checkpoint_metrics.csv"),
        optional=True,
    )


def _csv_fieldnames(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = list(csv.DictReader(handle).fieldnames or [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseError(f"cannot read CSV {path}: {exc}") from exc
    if not fields:
        raise ReleaseError(f"CSV has no header: {path}")
    return fields


def _stable_csv(path: Path, rows: Sequence[Mapping[str, object]], *, fields: Sequence[str] | None = None) -> None:
    """Write a deterministic CSV while preserving all source columns."""

    columns = list(fields or sorted({str(field) for row in rows for field in row}))
    if not columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: json.dumps(dict(item), sort_keys=True, default=str)):
            writer.writerow({field: row.get(field, "") for field in columns})
    os.replace(temporary, path)


def _canonical_source_csv(source: Path, destination: Path) -> None:
    fields = _csv_fieldnames(source)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseError(f"cannot read CSV {source}: {exc}") from exc
    _stable_csv(destination, rows, fields=sorted(fields))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tag_rows(rows: Iterable[Mapping[str, object]], experiment: str) -> list[dict[str, object]]:
    tagged: list[dict[str, object]] = []
    for row in rows:
        value = dict(row)
        value.setdefault("experiment", experiment)
        tagged.append(value)
    return tagged


def _rows(bundle: VerifiedBundle, table: str) -> list[dict[str, str]]:
    from lean_reward_hacking.schemas import load_table

    return load_table(bundle.path, table)


def _registered_analysis_settings(experiment: str) -> dict[str, object]:
    """Read the frozen label and statistics settings for one experiment."""

    import tomllib

    config_name = {
        "toy_fixed": "toy_colab.toml",
        "generic_mlp": "generic_colab.toml",
        "red_token_lm": "lm_colab.toml",
    }.get(experiment)
    payload: Mapping[str, object] = {}
    if config_name is not None:
        path = PROJECT_ROOT / "configs" / config_name
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload.get("labels"), Mapping) else {}
    statistics = (
        payload.get("statistics")
        if isinstance(payload.get("statistics"), Mapping)
        else {}
    )
    return {
        "c_on_min": float(labels.get("c_on_min", 0.95)),
        "invariant_c_off_min": float(labels.get("invariant_c_off_min", 0.90)),
        "strategic_c_off_max": float(labels.get("strategic_c_off_max", 0.10)),
        "dip_bootstrap": int(statistics.get("dip_bootstrap", 2000)),
        "mixture_bootstrap_replicates": int(statistics.get("mixture_bootstrap", 2000)),
        "bootstrap_seed": int(statistics.get("bootstrap_seed", 8675309)),
        "alpha": float(statistics.get("alpha", 0.05)),
        "minimum_component_weight": float(statistics.get("minimum_component_weight", 0.20)),
        "minimum_raw_c_off_separation": float(
            statistics.get("minimum_raw_c_off_separation", 0.30)
        ),
    }


def _finals(bundle: VerifiedBundle) -> list[dict[str, object]]:
    from lean_reward_hacking.analysis import load_final_rows

    settings = _registered_analysis_settings(bundle.spec.experiment)
    finals = load_final_rows(
        bundle.path,
        c_on_min=settings["c_on_min"],
        invariant_c_off_min=settings["invariant_c_off_min"],
        strategic_c_off_max=settings["strategic_c_off_max"],
    )
    run_rows = _rows(bundle, "runs.csv")
    if run_rows:
        complete_ids = {
            str(row.get("run_id") or "").strip()
            for row in run_rows
            if str(row.get("status") or "").strip().lower() in {"complete", "completed", "ok"}
        }
        finals = [
            row for row in finals
            if str(row.get("run_id") or "").strip() in complete_ids
        ]
    return _tag_rows(finals, bundle.spec.experiment)


def _complete_run_ids(bundle: VerifiedBundle) -> set[str]:
    return {
        str(row.get("run_id") or "").strip()
        for row in _rows(bundle, "runs.csv")
        if str(row.get("status") or "").strip().lower() in {"complete", "completed", "ok"}
        and str(row.get("run_id") or "").strip()
    }


def _identity_status(bundle: VerifiedBundle) -> tuple[bool, str]:
    """Check source-run identity fields before allowing primary claims."""

    runs = [
        row for row in _rows(bundle, "runs.csv")
        if str(row.get("status") or "").strip().lower() in {"complete", "completed", "ok"}
    ]
    if not runs:
        return False, "no complete source runs"
    missing = sorted(
        field for field in IDENTITY_FIELDS
        if any(not str(row.get(field) or "").strip() for row in runs)
    )
    conflicting = sorted(
        field for field in IDENTITY_FIELDS
        if len({str(row.get(field)) for row in runs if row.get(field) not in (None, "")}) > 1
    )
    if missing or conflicting:
        detail = {"missing": missing, "conflicting": conflicting, "complete_runs": len(runs)}
        return False, json.dumps(detail, sort_keys=True, separators=(",", ":"))
    return True, f"{len(runs)} complete source runs share all registered identity fields"


def _duplicate_run_check(records: Mapping[str, VerifiedBundle]) -> None:
    seen: dict[str, str] = {}
    for record in records.values():
        for row in _rows(record, "runs.csv"):
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            previous = seen.get(run_id)
            if previous is not None and previous != record.spec.experiment:
                raise ReleaseError(
                    f"run id {run_id!r} appears in experiments {previous!r} and {record.spec.experiment!r}"
                )
            seen[run_id] = record.spec.experiment


def _package_ready_lm() -> dict[str, object]:
    """Read the checked-in LM resource account without downloading weights."""

    from lean_reward_hacking.lm_training import LMTrainingConfig

    config_path = PROJECT_ROOT / "configs" / "lm_colab.toml"
    account_path = PROJECT_ROOT / "reports" / "LM_RESOURCE_REQUIREMENTS.json"
    if not config_path.is_file() or not account_path.is_file():
        return {
            "status": "package-unavailable",
            "reason": "LM config or resource account is absent",
        }
    try:
        account = _read_json_object(account_path)
        config = LMTrainingConfig.from_toml(config_path)
    except (OSError, UnicodeError, ValueError, ReleaseError) as exc:
        return {"status": "package-unavailable", "reason": f"invalid LM package account: {exc}"}
    mismatches: list[str] = []
    if account.get("config_file_sha256") != _sha256_file(config_path):
        mismatches.append("config_file_sha256")
    if account.get("training_config_sha256") != config.config_sha256:
        mismatches.append("training_config_sha256")
    requirements_path = PROJECT_ROOT / "requirements-lm-colab.txt"
    if (
        not requirements_path.is_file()
        or account.get("requirements_lm_colab_sha256") != _sha256_file(requirements_path)
    ):
        mismatches.append("requirements_lm_colab_sha256")
    required = (
        "model_id", "model_revision", "tokenizer_revision", "replicas",
        "accelerator_allowlist", "minimum_gpu_memory_gib", "minimum_host_ram_gib",
        "minimum_drive_free_gib", "estimated_gpu_hours_without_contingency",
        "estimated_gpu_hours_maximum", "launch_environment", "launch_instruction",
        "launch_cell", "unfinished_live_run",
    )
    missing = [field for field in required if account.get(field) in (None, "", [])]
    if missing or mismatches or account.get("paid_compute_authorized") is not False:
        return {
            "status": "package-incomplete",
            "missing": missing,
            "identity_mismatches": mismatches,
            "config": str(config_path),
            "resource_account": str(account_path),
        }
    return {
        **account,
        "status": "package-ready",
        "outcome_claim": False,
        "config": str(config_path),
        "resource_account": str(account_path),
        "endpoint_steps": list(config.endpoint_steps),
    }


def _stats_for(bundle: VerifiedBundle) -> dict[str, object]:
    from lean_reward_hacking.analysis import analyze_final_rows

    settings = _registered_analysis_settings(bundle.spec.experiment)
    finals = _finals(bundle)
    stats = analyze_final_rows(
        finals,
        dip_bootstrap=int(settings["dip_bootstrap"]),
        mixture_bootstrap_replicates=int(settings["mixture_bootstrap_replicates"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
        alpha=float(settings["alpha"]),
        minimum_component_weight=float(settings["minimum_component_weight"]),
        minimum_raw_c_off_separation=float(settings["minimum_raw_c_off_separation"]),
    )
    stats["experiment_scope"] = bundle.spec.experiment
    stats["complete_run_filter_applied"] = True
    endpoint_metrics: dict[str, object] = {}
    for output_name, candidates in {
        "c_on": ("c_on",),
        "c_off": ("c_off",),
        "gap": ("gap",),
        "goal": ("goal_score", "goal"),
        "gate": ("gate_effect", "gate"),
    }.items():
        values: list[float] = []
        for row in finals:
            parsed: float | None = None
            for field in candidates:
                try:
                    candidate = float(row.get(field))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate):
                    parsed = candidate
                    break
            if parsed is not None:
                values.append(parsed)
        endpoint_metrics[output_name] = {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    stats["endpoint_metrics"] = endpoint_metrics
    return stats


def _claim_gate(
    records: Mapping[str, VerifiedBundle],
    *,
    lm: VerifiedBundle | None,
    toy_stats: Mapping[str, object],
) -> dict[str, object]:
    from lean_reward_hacking.report import evaluate_claim_ladder

    toy = records["toy"]
    generic = records["generic"]
    perturbation = records["perturbation"]
    source_finals = _finals(toy)
    source_run_ids = _complete_run_ids(toy)
    source_trajectories = _tag_rows(
        [
            row for row in _rows(toy, "checkpoint_metrics.csv")
            if str(row.get("run_id") or "").strip() in source_run_ids
        ],
        toy.spec.experiment,
    )
    perturbation_rows: list[dict[str, object]] = []
    source_runs = {
        str(row.get("run_id") or "").strip(): row
        for row in _rows(toy, "runs.csv")
        if str(row.get("run_id") or "").strip() in source_run_ids
    }
    perturbation_identity_fields = {
        "source_config_sha256": "config_sha256",
        "source_archive_sha256": "source_archive_sha256",
        "train_dataset_sha256": "train_dataset_sha256",
        "eval_dataset_sha256": "eval_dataset_sha256",
        "reward_sha256": "reward_sha256",
        "objective_sha256": "objective_sha256",
    }
    rejected_perturbation_rows = 0
    for row in _rows(perturbation, "perturbation_trajectory.csv"):
        source_id = str(row.get("source_run_id") or "").strip()
        source_run = source_runs.get(source_id)
        if source_run is None or str(row.get("experiment") or "") != "toy_perturbation":
            rejected_perturbation_rows += 1
            continue
        if any(
            not _is_sha256(row.get(branch_field))
            or str(row.get(branch_field)) != str(source_run.get(source_field))
            for branch_field, source_field in perturbation_identity_fields.items()
        ):
            rejected_perturbation_rows += 1
            continue
        perturbation_rows.append(dict(row))
    perturbation_rows = _tag_rows(perturbation_rows, perturbation.spec.experiment)
    all_records: dict[str, VerifiedBundle] = dict(records)
    if lm is not None:
        all_records["lm"] = lm
    _duplicate_run_check(all_records)
    runs: list[dict[str, object]] = []
    for record in all_records.values():
        runs.extend(_tag_rows(_rows(record, "runs.csv"), record.spec.experiment))
    transfer_finals = _finals(generic)
    transfer_trajectories = _tag_rows(_rows(generic, "checkpoint_metrics.csv"), generic.spec.experiment)
    transfer_perturbations: list[dict[str, object]] = []
    if lm is not None:
        transfer_finals.extend(_finals(lm))
        transfer_trajectories.extend(_tag_rows(_rows(lm, "checkpoint_metrics.csv"), lm.spec.experiment))
        transfer_perturbations.extend(_tag_rows(_rows(lm, "perturbation_trajectory.csv"), lm.spec.experiment))
    audit_controls = _tag_rows(_rows(generic, "audit_control.csv"), generic.spec.experiment)
    identity_ok, identity_detail = _identity_status(toy)
    gate = evaluate_claim_ladder(
        finals=source_finals,
        stats=toy_stats,
        trajectories=source_trajectories,
        perturbations=perturbation_rows,
        runs=runs,
        transfer_finals=transfer_finals,
        transfer_trajectories=transfer_trajectories,
        transfer_perturbations=transfer_perturbations,
        audit_controls=audit_controls,
        evidence_identity_ok=identity_ok,
        evidence_identity_detail=identity_detail,
        perturbation_identity_ok=rejected_perturbation_rows == 0,
        perturbation_identity_detail=(
            f"rejected_identity_mismatched_rows={rejected_perturbation_rows}"
        ),
    )
    return gate


def _label_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label") or row.get("final_label") or "intermediate")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _write_derived_tables(
    output: Path,
    records: Mapping[str, VerifiedBundle],
    stats: Mapping[str, Mapping[str, object]],
    gate: Mapping[str, object],
) -> list[Path]:
    from lean_reward_hacking.analysis import write_final_summary_csv

    generated: list[Path] = []
    for key, record in records.items():
        destination = output / "tables" / key
        destination.mkdir(parents=True, exist_ok=True)
        for table_name in TABLE_NAMES:
            source = record.path / table_name
            if source.is_file():
                target = destination / table_name
                _canonical_source_csv(source, target)
                generated.append(target)
        finals_target = destination / "finals_normalized.csv"
        write_final_summary_csv(finals_target, _finals(record))
        generated.append(finals_target)
        stats_target = destination / "stats.json"
        _atomic_json(stats_target, stats[key])
        generated.append(stats_target)

    ladder_rows: list[dict[str, object]] = []
    levels = gate.get("levels")
    if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)):
        for item in levels:
            if isinstance(item, Mapping):
                ladder_rows.append(
                    {
                        "level": item.get("level", ""),
                        "claim": item.get("claim", ""),
                        "supported": item.get("supported", False),
                        "evidence": item.get("evidence", ""),
                        "falsifier": item.get("falsifier", ""),
                    }
                )
    ladder = output / "tables" / "claim_ladder.csv"
    _stable_csv(ladder, ladder_rows, fields=("level", "claim", "supported", "evidence", "falsifier"))
    generated.append(ladder)
    _atomic_json(output / "tables" / "claim_ladder.json", gate)
    generated.append(output / "tables" / "claim_ladder.json")
    summary = []
    for key, record in records.items():
        finals = _finals(record)
        summary.append(
            {
                "bundle": key,
                "experiment": record.spec.experiment,
                "n_final": len(finals),
                "label_counts_json": json.dumps(_label_counts(finals), sort_keys=True, separators=(",", ":")),
                "manifest_sha256": record.manifest_sha256,
                "source_archive_sha256": record.source_archive_sha256,
            }
        )
    summary_path = output / "tables" / "bundle_summary.csv"
    _stable_csv(summary_path, summary)
    generated.append(summary_path)
    return generated


def _write_claim_evidence_ledger(
    output: Path,
    records: Mapping[str, VerifiedBundle],
    gate: Mapping[str, object],
) -> Path:
    levels = {
        int(item.get("level", 0)): item
        for item in gate.get("levels", [])
        if isinstance(item, Mapping)
    }
    specifications = [
        (1, "toy", "finals_normalized.csv", "complete run at registered primary T; metric-derived label"),
        (2, "toy", "stats.json", "registered logit(C_off) two-component versus one-component test"),
        (3, "toy", "checkpoint_metrics.csv", "source label retained at registered 2T and 4T"),
        (4, "perturbation", "perturbation_trajectory.csv", "identity-bound primary recovery cells and matched controls"),
        (5, "generic", "finals_normalized.csv", "complete generic primary endpoints"),
        (5, "generic", "audit_control.csv", "paired audit cue, feature ablation, and paired-input controls"),
    ]
    if "lm" in records:
        specifications.extend(
            [
                (5, "lm", "finals_normalized.csv", "complete empirical LM primary endpoints"),
                (5, "lm", "perturbation_trajectory.csv", "empirical LM continuation and recovery branches"),
            ]
        )
    rows: list[dict[str, object]] = []
    for level, bundle, table, selector in specifications:
        item = levels.get(level, {})
        artifact = output / "tables" / bundle / table
        rows.append(
            {
                "level": level,
                "claim": item.get("claim", ""),
                "supported": bool(item.get("supported", False)),
                "source_bundle": bundle,
                "source_table": table,
                "selector": selector,
                "artifact": _artifact_locator(artifact) if artifact.is_file() else "absent",
                "sha256": _sha256_file(artifact) if artifact.is_file() else "absent",
            }
        )
    if "lm" not in records:
        item = levels.get(5, {})
        rows.append(
            {
                "level": 5,
                "claim": item.get("claim", ""),
                "supported": False,
                "source_bundle": "lm",
                "source_table": "empirical compact bundle",
                "selector": "required empirical LM evidence",
                "artifact": "absent",
                "sha256": "absent",
            }
        )
    path = output / "tables" / "claim_evidence.csv"
    _stable_csv(
        path,
        rows,
        fields=(
            "level", "claim", "supported", "source_bundle", "source_table",
            "selector", "artifact", "sha256",
        ),
    )
    return path


def _regenerate_figures(output: Path, records: Mapping[str, VerifiedBundle]) -> list[Path]:
    from lean_reward_hacking.plotting import FIGURE_SPECS, plot_all

    expected = {filename for filename, _ in FIGURE_SPECS}
    if expected != set(FIGURE_NAMES):
        raise ReleaseError("registered figure list drifted from plotting module")
    available_keys = set(records)
    missing_sources = sorted(set(FIGURE_SOURCE_KEYS.values()) - available_keys)
    if missing_sources:
        raise ReleaseError(
            "figure regeneration is missing source bundle(s): " + ", ".join(missing_sources)
        )
    generated: list[Path] = []
    figure_dir = output / "figures"
    for filename, kind in FIGURE_SPECS:
        key = FIGURE_SOURCE_KEYS.get(kind)
        if key is None:
            raise ReleaseError(f"no registered source bundle for figure kind {kind!r}")
        rendered = plot_all(records[key].path, figure_dir, figures=[kind], strict=True)
        rendered_names = {path.name for path in rendered}
        if filename not in rendered_names:
            raise ReleaseError(f"figure regeneration for {key} omitted {filename}")
        path = figure_dir / filename
        generated.append(path)
        sidecar = path.with_suffix(".metadata.json")
        if not sidecar.is_file():
            raise ReleaseError(f"figure metadata sidecar is missing: {sidecar}")
        generated.append(sidecar)
    return generated


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _unfinished_rows(records: Mapping[str, VerifiedBundle]) -> list[dict[str, object]]:
    unfinished: list[dict[str, object]] = []
    for key, record in records.items():
        for row in _rows(record, "runs.csv"):
            status = str(row.get("status") or "").strip().lower()
            if status in {"complete", "completed", "ok"}:
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            config_name = {
                "toy_fixed": "toy_colab.toml",
                "generic_mlp": "generic_colab.toml",
                "toy_basin": "basin_colab.toml",
                "toy_perturbation": "perturbation_colab.toml",
                LM_EXPERIMENT: "lm_colab.toml",
            }.get(record.spec.experiment)
            last_checkpoint = (
                row.get("last_checkpoint") or row.get("final_step") or row.get("completed_step")
            )
            last_marker = row.get("marker_path") or record.provenance.get("last_marker")
            remote_raw_uri = row.get("remote_raw_uri") or record.provenance.get("remote_raw_uri")
            restart_command = row.get("restart_command") or record.provenance.get("restart_command")
            resource = row.get("resource_requirement") or record.provenance.get(
                "resource_requirement"
            )
            missing = [
                name
                for name, value in (
                    ("last_checkpoint", last_checkpoint),
                    ("last_marker", last_marker),
                    ("remote_raw_uri", remote_raw_uri),
                    ("resource_requirement", resource),
                    ("restart_command", restart_command),
                    ("config", config_name),
                )
                if value in (None, "", {})
            ]
            if missing:
                raise ReleaseError(
                    f"unfinished run {run_id!r} lacks exact restart accounting: "
                    + ", ".join(missing)
                )
            unfinished.append(
                {
                    "experiment": record.spec.experiment,
                    "bundle": key,
                    "run_id": run_id,
                    "status": status or "missing",
                    "last_checkpoint": last_checkpoint,
                    "last_marker": last_marker,
                    "remote_raw_uri": remote_raw_uri,
                    "resource": (
                        resource
                        if isinstance(resource, str)
                        else json.dumps(resource, sort_keys=True, separators=(",", ":"))
                    ),
                    "restart_command": restart_command,
                }
            )
    return unfinished


def _package_unfinished_row(lm_package: Mapping[str, object]) -> dict[str, object]:
    status = str(lm_package.get("status") or "package-unavailable")
    if status != "package-ready":
        raise ReleaseError(
            "LM package has no empirical bundle and is not launch-ready: "
            + json.dumps(dict(lm_package), sort_keys=True, default=str)
        )
    launch_cell = str(lm_package.get("launch_cell") or "").strip()
    if not launch_cell:
        raise ReleaseError("LM resource account lacks an executable launch cell")
    return {
        "experiment": LM_EXPERIMENT,
        "bundle": "lm-package",
        "run_id": "unlaunched",
        "status": "package-ready-unlaunched",
        "last_checkpoint": "none; no live run started",
        "last_marker": "none; no live run started",
        "remote_raw_uri": "/content/drive/MyDrive/two_equilibria/v1/runs/red_token_lm/<RUN_ID>",
        "resource": _compact_json(
            {
                "accelerator_allowlist": lm_package.get("accelerator_allowlist"),
                "minimum_gpu_memory_gib": lm_package.get("minimum_gpu_memory_gib"),
                "minimum_host_ram_gib": lm_package.get("minimum_host_ram_gib"),
                "minimum_drive_free_gib": lm_package.get("minimum_drive_free_gib"),
                "estimated_gpu_hours_without_contingency": lm_package.get(
                    "estimated_gpu_hours_without_contingency"
                ),
                "estimated_gpu_hours_maximum": lm_package.get(
                    "estimated_gpu_hours_maximum"
                ),
            }
        ),
        "restart_command": launch_cell + "; then run notebooks/05_lm_workflow_colab.ipynb",
    }


def _write_unfinished_runs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _stable_csv(
        path,
        rows,
        fields=(
            "experiment", "bundle", "run_id", "status", "last_checkpoint",
            "last_marker", "remote_raw_uri", "resource", "restart_command",
        ),
    )


def _nested(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _artifact_locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _statistical_evidence_rows(
    stats: Mapping[str, Mapping[str, object]],
    output: Path,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for key, value in stats.items():
        primary = _nested(value, "bimodality", "primary")
        labels = _nested(value, "labels", "probabilities")
        stats_path = output / "tables" / key / "stats.json"
        rows.append(
            (
                key,
                _nested(value, "sample", "n_independent_final_runs"),
                _compact_json(labels or {}),
                _compact_json(_nested(value, "endpoint_metrics", "c_on") or {}),
                _compact_json(_nested(value, "endpoint_metrics", "c_off") or {}),
                _compact_json(_nested(value, "endpoint_metrics", "gap") or {}),
                _compact_json(_nested(value, "endpoint_metrics", "goal") or {}),
                _compact_json(_nested(value, "endpoint_metrics", "gate") or {}),
                _nested(primary, "bic_delta") if isinstance(primary, Mapping) else None,
                _nested(primary, "p_value") if isinstance(primary, Mapping) else None,
                (
                    _nested(primary, "raw_c_off_component_mean_separation")
                    if isinstance(primary, Mapping) else None
                ),
                _nested(value, "bimodality", "dip", "p_value"),
                _nested(value, "bimodality", "mixture", "successful_replicates"),
                _nested(value, "configuration", "mixture_bootstrap_replicates"),
                _artifact_locator(stats_path),
                _sha256_file(stats_path),
            )
        )
    return rows


def _figure_evidence_rows(
    output: Path,
    records: Mapping[str, VerifiedBundle],
) -> list[tuple[object, ...]]:
    figure_dir = output / "figures"
    rows: list[tuple[object, ...]] = []
    for filename in FIGURE_NAMES:
        kind = next(
            kind for kind, registered_name in {
                "final_gap": "fig01_final_gap_histogram.svg",
                "trajectories": "fig02_training_trajectories.svg",
                "basin": "fig03_basin_phase_diagram.svg",
                "perturbation": "fig04_perturbation_recovery.svg",
                "reward": "fig05_reward_vs_hidden_misalignment.svg",
                "control": "fig06_control_audit_swaps.svg",
            }.items() if registered_name == filename
        )
        source = records[FIGURE_SOURCE_KEYS[kind]]
        figure = figure_dir / filename
        sidecar = figure.with_suffix(".metadata.json")
        rows.append(
            (
                filename,
                source.spec.experiment,
                _artifact_locator(figure),
                _sha256_file(figure),
                _sha256_file(sidecar),
            )
        )
    return rows


def _report_text(
    *,
    records: Mapping[str, VerifiedBundle],
    optional_lm: VerifiedBundle | None,
    lm_package: Mapping[str, object],
    stats: Mapping[str, Mapping[str, object]],
    gate: Mapping[str, object],
    output: Path,
    unfinished_rows: Sequence[Mapping[str, object]],
) -> str:
    report_records = dict(records)
    if optional_lm is not None:
        report_records["lm"] = optional_lm
    observation_rows = []
    for key, record in report_records.items():
        finals = _finals(record)
        observation_rows.append(
            (
                key,
                record.spec.experiment,
                len(finals),
                json.dumps(_label_counts(finals), sort_keys=True, separators=(",", ":")),
                len(_rows(record, "checkpoint_metrics.csv")),
                len(_rows(record, "perturbation_trajectory.csv")),
                len(_rows(record, "basin_cells.csv")),
            )
        )
    levels = gate.get("levels", [])
    level_rows = []
    falsifiers = []
    if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)):
        for item in levels:
            if not isinstance(item, Mapping):
                continue
            level_rows.append(
                (
                    item.get("level", ""),
                    item.get("claim", ""),
                    "supported" if item.get("supported") else "not supported",
                    item.get("evidence", ""),
                )
            )
            falsifiers.append((item.get("level", ""), item.get("falsifier", "")))
    unfinished = list(unfinished_rows)
    report_lines = [
        "# Two-persona RLHF equilibria final report",
        "",
        "This report is generated from verified imported compact bundles. Raw checkpoints and logs remain in their recorded remote locations.",
        "",
        "## Observations",
        "",
        _markdown_table(
            ("Bundle", "Experiment", "Independent finals", "Labels", "Checkpoint rows", "Perturbation rows", "Basin cells"),
            observation_rows,
        ),
        "",
        "The toy statistical summary is regenerated from the saved final rows. Its registered primary statistic is logit(C_off); secondary gap-scale diagnostics remain in the bundle statistics.",
        "",
        "## Registered statistical evidence",
        "",
        _markdown_table(
            (
                "Bundle", "Independent finals", "Label proportions and Wilson 95%",
                "C_on summary", "C_off summary", "Gap summary", "Goal summary", "Gate summary",
                "Mixture ΔBIC", "Bootstrap p", "Raw C_off separation", "Gap dip p",
                "Successful bootstrap", "Registered bootstrap", "Saved statistics",
                "Statistics SHA-256",
            ),
            _statistical_evidence_rows(stats, output),
        ),
        "",
        "Primary rows come from the declared T checkpoint. The saved tables retain every continuous C_on, C_off, gap, reward, goal, and gate field supplied by each architecture. Threshold sensitivity, continuation, paired counts, and recovery rows remain in the release tables.",
        "",
        "## Inferences",
        "",
        f"Strongest supported claim: {gate.get('strongest_claim', 'unavailable')}.",
        f"Strongest supported level: {gate.get('strongest_level', 0)}.",
        f"Attractor wording permitted: {bool(gate.get('phrase_two_rlhf_attractors_allowed', False))}.",
        "",
        "The inference follows the registered claim gate. Continuous scores, endpoint labels, continuation rows, and source-conditioned branches remain separate evidence channels.",
        "",
        "## Limitations",
        "",
        f"Language-model package status: {lm_package.get('status', 'unknown')}.",
        "LM outcome claims require an empirical red_token_lm bundle with completed rows; package-ready status carries no outcome claim.",
        f"Optional empirical LM bundle imported: {optional_lm is not None}.",
        f"Unfinished live run records: {len(unfinished)}.",
        "",
        "## Falsifiers",
        "",
        _markdown_table(("Level", "Registered falsifier"), falsifiers),
        "",
        "A shared endpoint under continuation, failed source-conditioned recovery, missing identity binding, or a failed registered modality criterion keeps the corresponding claim level closed.",
        "",
        "## Claim ladder",
        "",
        _markdown_table(("Level", "Claim", "Status", "Evidence"), level_rows),
        "",
        (
            f"Claim evidence ledger: `{_artifact_locator(output / 'tables' / 'claim_evidence.csv')}` "
            f"(SHA-256 `{_sha256_file(output / 'tables' / 'claim_evidence.csv')}`)."
        ),
        "",
        "## Provenance",
        "",
        _markdown_table(
            ("Bundle", "Path", "Manifest SHA-256", "Source archive SHA-256"),
            [
                (key, record.path, record.manifest_sha256, record.source_archive_sha256)
                for key, record in records.items()
            ]
            + (
                [("lm", optional_lm.path, optional_lm.manifest_sha256, optional_lm.source_archive_sha256)]
                if optional_lm is not None
                else []
            ),
        ),
        "",
        f"Toy stats configuration: {json.dumps(stats['toy'].get('configuration', {}), sort_keys=True, separators=(',', ':'))}.",
        f"Generated release tree: {output}.",
        "Literature boundary: reports/literature_gap_audit.md.",
        "",
        "## Figure evidence",
        "",
        _markdown_table(
            ("Figure", "Source experiment", "Artifact", "Figure SHA-256", "Sidecar SHA-256"),
            _figure_evidence_rows(output, records),
        ),
        "",
        "Every figure is regenerated from one registered source bundle. Its metadata sidecar records the table inputs and plot configuration.",
        "",
        "## Audit evidence",
        "",
        f"The machine-verifiable four-pass ledger is `{_artifact_locator(output / 'report_audit.json')}`. It records literature-source coverage, statistical checks, claim-scope checks, artifact hashes, and rendered-review status.",
        "",
        "## Unfinished runs and resource account",
        "",
    ]
    if unfinished:
        report_lines.append(
            _markdown_table(
                ("Experiment", "Run", "Status", "Last checkpoint", "Last marker", "Remote path", "Resource", "Restart command"),
                [
                    (
                        row["experiment"],
                        row["run_id"],
                        row["status"],
                        row["last_checkpoint"],
                        row["last_marker"],
                        row["remote_raw_uri"],
                        row["resource"],
                        row["restart_command"],
                    )
                    for row in unfinished
                ],
            )
        )
    else:
        report_lines.append("No unfinished live run records were present in the imported compact bundles.")
    report_lines.extend(
        [
            "",
            "LM resource account from reports/LM_RESOURCE_REQUIREMENTS.json:",
            "",
            _markdown_table(
                ("Status", "Model", "Revision", "Accelerator", "Host RAM GiB", "GPU hours base", "GPU hours max", "Drive GiB"),
                [
                    (
                        lm_package.get("status", "unknown"),
                        lm_package.get("model_id", "unrecorded"),
                        lm_package.get("model_revision", "unrecorded"),
                        json.dumps(
                            lm_package.get("accelerator_allowlist", "unrecorded"),
                            separators=(",", ":"),
                        ),
                        lm_package.get("minimum_host_ram_gib", "unrecorded"),
                        lm_package.get("estimated_gpu_hours_without_contingency", "unrecorded"),
                        lm_package.get("estimated_gpu_hours_maximum", "unrecorded"),
                        lm_package.get("minimum_drive_free_gib", "unrecorded"),
                    )
                ],
            ),
            "",
        ]
    )
    return "\n".join(report_lines)


def _write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _output_records(paths: Iterable[Path], root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        if path.is_file():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = str(path)
            records.append({"path": relative, "sha256": _sha256_file(path)})
    return records


def _report_audit_payload(
    *,
    report_path: Path,
    report_text: str,
    generated: Sequence[Path],
    destination: Path,
    stats: Mapping[str, Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict[str, object]:
    source_ledger = PROJECT_ROOT / "reports" / "source_ledger.csv"
    literature_audit = PROJECT_ROOT / "reports" / "literature_gap_audit.md"
    source_rows: list[dict[str, str]] = []
    if source_ledger.is_file():
        with source_ledger.open("r", encoding="utf-8", newline="") as handle:
            source_rows = [dict(row) for row in csv.DictReader(handle)]
    exact_locator_rows = sum(
        bool(
            str(
                row.get("exact_locators")
                or row.get("exact_locator")
                or row.get("locator")
                or ""
            ).strip()
        )
        for row in source_rows
    )
    literature_passed = bool(
        literature_audit.is_file()
        and source_rows
        and exact_locator_rows == len(source_rows)
    )
    stats_paths = [destination / "tables" / key / "stats.json" for key in stats]
    statistics_passed = bool(
        stats_paths
        and all(path.is_file() for path in stats_paths)
        and all(
            isinstance(value.get("configuration"), Mapping)
            for value in stats.values()
        )
    )
    required_headings = (
        "## Observations", "## Registered statistical evidence", "## Inferences",
        "## Limitations", "## Falsifiers", "## Claim ladder", "## Provenance",
        "## Figure evidence", "## Audit evidence", "## Unfinished runs and resource account",
    )
    claim_scope_passed = all(heading in report_text for heading in required_headings)
    if not bool(gate.get("phrase_two_rlhf_attractors_allowed", False)):
        claim_scope_passed = claim_scope_passed and "two RLHF attractors" not in report_text
    figure_paths = [destination / "figures" / name for name in FIGURE_NAMES]
    sidecars = [path.with_suffix(".metadata.json") for path in figure_paths]
    artifact_passed = all(path.is_file() for path in [*figure_paths, *sidecars])
    return {
        "schema_version": 1,
        "status": "pending_rendered_review",
        "overall_passed": False,
        "passes": [
            {
                "pass": 1,
                "name": "claim_and_source",
                "status": "passed" if literature_passed else "failed",
                "result": "PASS" if literature_passed else "FAIL",
                "input_snapshot": {
                    "report_sha256": _sha256_file(report_path),
                    "literature_audit_sha256": (
                        _sha256_file(literature_audit) if literature_audit.is_file() else None
                    ),
                    "source_ledger_sha256": (
                        _sha256_file(source_ledger) if source_ledger.is_file() else None
                    ),
                },
                "full_scope_read": [
                    _artifact_locator(report_path),
                    _artifact_locator(literature_audit),
                    _artifact_locator(source_ledger),
                ],
                "locator_findings": {
                    "source_rows": len(source_rows),
                    "rows_with_exact_locators": exact_locator_rows,
                },
                "edits": ["generated source-bound claim and provenance sections"],
                "checks": ["all literature rows carry exact locators", "literature audit exists"],
                "unresolved": [] if literature_passed else ["literature source coverage failed"],
                "evidence": {
                    "literature_audit": str(literature_audit),
                    "literature_audit_sha256": (
                        _sha256_file(literature_audit) if literature_audit.is_file() else None
                    ),
                    "source_ledger": str(source_ledger),
                    "source_ledger_sha256": (
                        _sha256_file(source_ledger) if source_ledger.is_file() else None
                    ),
                    "source_rows": len(source_rows),
                    "rows_with_exact_locators": exact_locator_rows,
                },
            },
            {
                "pass": 2,
                "name": "equation_and_statistics",
                "status": "passed" if statistics_passed else "failed",
                "result": "PASS" if statistics_passed else "FAIL",
                "input_snapshot": {
                    "report_sha256": _sha256_file(report_path),
                    "registered_methods_sha256": _sha256_file(
                        PROJECT_ROOT / "reports" / "statistical_methods.md"
                    ),
                },
                "full_scope_read": [
                    _artifact_locator(PROJECT_ROOT / "reports" / "statistical_methods.md"),
                    *[_artifact_locator(path) for path in stats_paths],
                ],
                "locator_findings": {
                    "statistics_files": len(stats_paths),
                    "registered_configuration_records": sum(
                        isinstance(value.get("configuration"), Mapping)
                        for value in stats.values()
                    ),
                },
                "edits": ["regenerated complete-run statistics and quantitative evidence table"],
                "checks": ["registered settings recorded", "statistics artifacts hashable"],
                "unresolved": [] if statistics_passed else ["registered statistics verification failed"],
                "evidence": {
                    "statistic_files": [
                        {"path": str(path), "sha256": _sha256_file(path)}
                        for path in stats_paths if path.is_file()
                    ],
                    "registered_methods_sha256": _sha256_file(
                        PROJECT_ROOT / "reports" / "statistical_methods.md"
                    ),
                },
            },
            {
                "pass": 3,
                "name": "prose_and_claim_scope",
                "status": "passed" if claim_scope_passed else "failed",
                "result": "PASS" if claim_scope_passed else "FAIL",
                "input_snapshot": {"report_sha256": _sha256_file(report_path)},
                "full_scope_read": [_artifact_locator(report_path)],
                "locator_findings": {
                    "required_headings": list(required_headings),
                    "present_headings": [heading for heading in required_headings if heading in report_text],
                },
                "edits": ["generated separated observation, inference, limitation, and falsifier sections"],
                "checks": ["claim ladder applied", "attractor phrase gate applied"],
                "unresolved": [] if claim_scope_passed else ["report claim-scope check failed"],
                "evidence": {
                    "report": str(report_path),
                    "report_sha256": _sha256_file(report_path),
                    "required_headings": list(required_headings),
                    "strongest_level": gate.get("strongest_level", 0),
                    "strongest_claim": gate.get("strongest_claim", ""),
                    "attractor_phrase_allowed": gate.get(
                        "phrase_two_rlhf_attractors_allowed", False
                    ),
                },
            },
            {
                "pass": 4,
                "name": "artifact_and_render",
                "status": "pending_rendered_review" if artifact_passed else "failed",
                "result": "PENDING" if artifact_passed else "FAIL",
                "input_snapshot": {"report_sha256": _sha256_file(report_path)},
                "full_scope_read": [
                    *[_artifact_locator(path) for path in figure_paths],
                    *[_artifact_locator(path) for path in sidecars],
                ],
                "locator_findings": {"figures": len(figure_paths), "sidecars": len(sidecars)},
                "edits": ["regenerated six figures and metadata sidecars"],
                "checks": ["all figure files exist", "all sidecars exist"],
                "unresolved": ["rendered visual review pending"] if artifact_passed else ["machine artifact check failed"],
                "evidence": {
                    "machine_artifacts_passed": artifact_passed,
                    "artifacts": _output_records([*figure_paths, *sidecars], destination),
                    "rendered_review": None,
                },
            },
        ],
        "generated_outputs": _output_records(generated, destination),
    }


def build_release(
    *,
    compact_root: str | Path = PROJECT_ROOT / "results" / "compact",
    output: str | Path = PROJECT_ROOT / "results" / "release",
    report: str | Path = PROJECT_ROOT / "reports" / "final_report.md",
    toy_bundle: str | Path | None = None,
    generic_bundle: str | Path | None = None,
    basin_bundle: str | Path | None = None,
    perturbation_bundle: str | Path | None = None,
    lm_bundle: str | Path | None = None,
) -> dict[str, object]:
    """Verify inputs, regenerate outputs, and write the final report."""

    records = resolve_required_bundles(
        compact_root,
        explicit={
            "toy": toy_bundle,
            "generic": generic_bundle,
            "basin": basin_bundle,
            "perturbation": perturbation_bundle,
        },
    )
    optional_lm = _optional_lm_bundle(lm_bundle)
    all_records = dict(records)
    if optional_lm is not None:
        all_records["lm"] = optional_lm
    stats = {key: _stats_for(record) for key, record in all_records.items()}
    toy_stats = stats["toy"]
    gate = _claim_gate(records, lm=optional_lm, toy_stats=toy_stats)
    package_status = _package_ready_lm()
    unfinished_rows = _unfinished_rows(all_records)
    if optional_lm is None:
        unfinished_rows.append(_package_unfinished_row(package_status))

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    generated = _write_derived_tables(destination, all_records, stats, gate)
    claim_evidence_path = _write_claim_evidence_ledger(destination, all_records, gate)
    generated.append(claim_evidence_path)
    unfinished_path = destination / "tables" / "unfinished_runs.csv"
    _write_unfinished_runs(unfinished_path, unfinished_rows)
    generated.append(unfinished_path)
    generated.extend(_regenerate_figures(destination, all_records))
    report_path = Path(report)
    report_text = _report_text(
        records=records,
        optional_lm=optional_lm,
        lm_package=package_status,
        stats=stats,
        gate=gate,
        output=destination,
        unfinished_rows=unfinished_rows,
    )
    _write_report(report_path, report_text)
    generated.append(report_path)
    audit_path = destination / "report_audit.json"
    _atomic_json(
        audit_path,
        _report_audit_payload(
            report_path=report_path,
            report_text=report_text,
            generated=generated,
            destination=destination,
            stats=stats,
            gate=gate,
        ),
    )
    generated.append(audit_path)
    release_manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "required_bundles": {
            key: {
                "experiment": record.spec.experiment,
                "path": str(record.path),
                "manifest_sha256": record.manifest_sha256,
                "source_archive_sha256": record.source_archive_sha256,
            }
            for key, record in records.items()
        },
        "optional_lm_bundle": (
            {
                "path": str(optional_lm.path),
                "manifest_sha256": optional_lm.manifest_sha256,
                "source_archive_sha256": optional_lm.source_archive_sha256,
            }
            if optional_lm is not None
            else None
        ),
        "lm_package": package_status,
        "claim_gate": {
            "strongest_level": gate.get("strongest_level", 0),
            "strongest_claim": gate.get("strongest_claim", ""),
            "phrase_two_rlhf_attractors_allowed": gate.get("phrase_two_rlhf_attractors_allowed", False),
        },
        "unfinished_runs": [dict(row) for row in unfinished_rows],
        "figure_sources": {
            filename: records[FIGURE_SOURCE_KEYS[kind]].spec.experiment
            for filename, kind in (
                ("fig01_final_gap_histogram.svg", "final_gap"),
                ("fig02_training_trajectories.svg", "trajectories"),
                ("fig03_basin_phase_diagram.svg", "basin"),
                ("fig04_perturbation_recovery.svg", "perturbation"),
                ("fig05_reward_vs_hidden_misalignment.svg", "reward"),
                ("fig06_control_audit_swaps.svg", "control"),
            )
        },
        "outputs": _output_records(generated, destination),
        "report": str(report_path),
        "report_audit": str(audit_path),
    }
    _atomic_json(destination / "release_manifest.json", release_manifest)
    return release_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release_pipeline.py")
    parser.add_argument("--compact-root", type=Path, default=PROJECT_ROOT / "results" / "compact")
    parser.add_argument("--output", "--out", dest="output", type=Path, default=PROJECT_ROOT / "results" / "release")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "final_report.md")
    parser.add_argument("--toy-bundle", type=Path)
    parser.add_argument("--generic-bundle", type=Path)
    parser.add_argument("--basin-bundle", type=Path)
    parser.add_argument("--perturbation-bundle", type=Path)
    parser.add_argument("--lm-bundle", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_release(
            compact_root=args.compact_root,
            output=args.output,
            report=args.report,
            toy_bundle=args.toy_bundle,
            generic_bundle=args.generic_bundle,
            basin_bundle=args.basin_bundle,
            perturbation_bundle=args.perturbation_bundle,
            lm_bundle=args.lm_bundle,
        )
    except (ReleaseError, OSError, ValueError) as exc:
        print(f"release failed: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"release complete: {args.output}")
        print(f"report: {args.report}")
        print(f"strongest level: {manifest['claim_gate']['strongest_level']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BundleSpec",
    "PIPELINE_VERSION",
    "ReleaseError",
    "REQUIRED_LIVE_BUNDLES",
    "VerifiedBundle",
    "build_release",
    "main",
    "resolve_required_bundles",
    "verify_imported_bundle",
]
