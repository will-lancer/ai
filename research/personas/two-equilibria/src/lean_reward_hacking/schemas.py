"""Compact result schemas and bundle validation.

The analysis package deliberately uses the Python standard library for bundle
validation.  Colab jobs may produce rich remote artefacts, while the local
bundle contains only small, tabular summaries and provenance metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_NAMES = frozenset({"bundle_manifest.json", "manifest.json"})
CHECKSUMS_NAME = "checksums.sha256"

TABLE_ALIASES = {
    "finals.csv": "final_summary.csv",
    "final_metrics.csv": "final_summary.csv",
    "trajectories.csv": "checkpoint_metrics.csv",
    "perturbations.csv": "perturbation_trajectory.csv",
    "basin.csv": "basin_cells.csv",
}

# The allowlist is intentionally flat.  Raw weights, checkpoints, logs, and
# large arrays belong in the remote run directory.
COMPACT_ALLOWLIST = frozenset(
    {
        ".gitkeep",
        "runs.csv",
        "pair_counts.csv",
        "checkpoint_metrics.csv",
        "final_summary.csv",
        "basin_cells.csv",
        "perturbation_trajectory.csv",
        "audit_control.csv",
        "threshold_sensitivity.csv",
        "stats.json",
        "provenance.json",
        "bundle_manifest.json",
        "manifest.json",
        CHECKSUMS_NAME,
        *TABLE_ALIASES.keys(),
    }
)

# Required fields are a conservative common denominator.  Producers may add
# columns from the per-table allowlist below as their experiments mature.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "runs.csv": ("run_id", "seed", "status"),
    "pair_counts.csv": ("run_id", "n_pairs", "n11", "n10", "n01", "n00"),
    "checkpoint_metrics.csv": ("run_id", "step", "c_on", "c_off", "gap"),
    "final_summary.csv": ("run_id", "c_on", "c_off", "gap"),
    "basin_cells.csv": ("harm_strength", "audit_sensitivity", "n_seeds"),
    "perturbation_trajectory.csv": (
        "source_run_id",
        "branch_id",
        "step_since_branch",
    ),
    "audit_control.csv": (
        "experiment",
        "run_id",
        "control_kind",
        "available",
    ),
    "threshold_sensitivity.csv": (
        "experiment",
        "threshold_set",
        "n_runs",
        "status",
    ),
}

KNOWN_COLUMNS: dict[str, frozenset[str]] = {
    "runs.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "condition",
            "run_id",
            "seed",
            "shuffle_seed",
            "config_id",
            "config_sha256",
            "git_commit",
            "status",
            "final_step",
            "final_label",
            "primary_step",
            "primary_label",
            "completed_step",
            "completed_label",
            "train_dataset_sha256",
            "eval_dataset_sha256",
            "reward_sha256",
            "objective_sha256",
            "source_archive_sha256",
            "threshold_set",
            "remote_raw_uri",
            "marker_path",
            "restart_command",
            "resource_requirement",
            "last_checkpoint",
            "started_at_utc",
            "completed_at_utc",
            "checkpoint_count",
            "retention_2T",
            "retention_4T",
            "continuation_2T_label",
            "continuation_4T_label",
            "continuation_retention_status",
            "continuation_retention_ok",
        }
    ),
    "pair_counts.csv": frozenset(
        {
            "schema_version",
            "run_id",
            "branch_id",
            "source_run_id",
            "step",
            "eval_variant",
            "eval_set_hash",
            "n11",
            "n10",
            "n01",
            "n00",
            "n_pairs",
            "primary_step",
            "completed_step",
        }
    ),
    "checkpoint_metrics.csv": frozenset(
        {
            "schema_version",
            "run_id",
            "branch_id",
            "source_run_id",
            "step",
            "eval_variant",
            "eval_set_hash",
            "c_on",
            "c_off",
            "gap",
            "goal_score",
            "goal",
            "gate_effect",
            "gate",
            "off_audit_logit",
            "train_reward",
            "reward",
            "hidden_misalignment_rate",
            "hidden_misalignment",
            "action_entropy",
            "train_loss",
            "n_pairs",
            "is_final",
            "is_primary",
            "is_terminal",
            "primary_step",
            "primary_label",
            "completed_step",
            "completed_label",
            "label",
        }
    ),
    "final_summary.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "condition",
            "run_id",
            "seed",
            "step",
            "c_on",
            "c_off",
            "gap",
            "goal_score",
            "goal",
            "gate_effect",
            "gate",
            "off_audit_logit",
            "train_reward",
            "reward",
            "hidden_misalignment_rate",
            "hidden_misalignment",
            "label",
            "final_label",
            "primary_step",
            "primary_label",
            "completed_step",
            "completed_label",
            "is_primary",
            "threshold_set",
            "n_pairs",
            "eval_set_hash",
            "retention_2T",
            "retention_4T",
            "continuation_2T_label",
            "continuation_4T_label",
            "continuation_retention_status",
            "continuation_retention_ok",
        }
    ),
    "basin_cells.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "harm_strength",
            "audit_sensitivity",
            "n_seeds",
            "n_complete",
            "n_invariant",
            "n_strategic",
            "n_intermediate",
            "p_invariant",
            "p_strategic",
            "p_intermediate",
            "ci_low",
            "ci_high",
            "config_sha256",
        }
    ),
    "perturbation_trajectory.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "config_sha256",
            "source_config_sha256",
            "source_archive_sha256",
            "train_dataset_sha256",
            "eval_dataset_sha256",
            "reward_sha256",
            "objective_sha256",
            "source_run_id",
            "source_step",
            "source_label",
            "branch_id",
            "branch_kind",
            "control_kind",
            "control_group",
            "optimizer_policy",
            "intervention",
            "strength",
            "branch_seed",
            "step_since_branch",
            "c_on",
            "c_off",
            "gap",
            "goal_score",
            "goal",
            "gate_effect",
            "gate",
            "off_audit_logit",
            "reward",
            "hidden_misalignment_rate",
            "hidden_misalignment",
            "d_source",
            "d_opposite",
            "source_label_retention",
            "source_distance_closer",
            "intervention_feasible",
            "intervention_status",
            "mode_label",
            "recovery_status",
        }
    ),
    "audit_control.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "run_id",
            "control_kind",
            "available",
            "reason",
            "baseline_c_on",
            "baseline_c_off",
            "baseline_gap",
            "control_c_on",
            "control_c_off",
            "control_gap",
            "delta_c_on",
            "delta_c_off",
            "delta_gap",
            "n_pairs",
            "eval_set_hash",
        }
    ),
    "threshold_sensitivity.csv": frozenset(
        {
            "schema_version",
            "experiment",
            "architecture",
            "threshold_set",
            "c_on_min",
            "invariant_c_off_min",
            "strategic_c_off_max",
            "require_goal_sign",
            "n_runs",
            "n_invariant",
            "n_strategic",
            "n_intermediate",
            "labels_json",
            "counts_json",
            "status",
        }
    ),
}


class BundleValidationError(ValueError):
    """Raised when a compact bundle violates the local result contract."""


def canonical_name(name: str) -> str:
    """Return the canonical table name for an accepted alias."""

    return TABLE_ALIASES.get(name, name)


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _relative_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            files.append(item.relative_to(root))
        elif item.is_symlink():
            # Symlinks make provenance ambiguous even when their target looks
            # harmless.  Treat them as invalid bundle members.
            files.append(item.relative_to(root))
    return files


def read_csv_rows(path: os.PathLike[str] | str) -> list[dict[str, str]]:
    """Read a UTF-8 CSV while preserving the producer's string values."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BundleValidationError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _validate_table(path: Path, canonical: str, *, strict: bool) -> list[str]:
    problems: list[str] = []
    endpoint_rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if not fields:
                return [f"{canonical}: missing CSV header"]
            if len(fields) != len(set(fields)):
                problems.append(f"{canonical}: duplicate CSV header")
            missing = [field for field in REQUIRED_COLUMNS.get(canonical, ()) if field not in fields]
            if missing:
                problems.append(f"{canonical}: missing columns {','.join(missing)}")
            if strict:
                unknown = sorted(set(fields) - KNOWN_COLUMNS.get(canonical, frozenset()))
                if unknown:
                    problems.append(f"{canonical}: unknown columns {','.join(unknown)}")
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    problems.append(f"{canonical}: extra fields at row {row_number}")
                    break
                if all(value in (None, "") for value in row.values()):
                    problems.append(f"{canonical}: blank row at {row_number}")
                    break
                if canonical in {
                    "runs.csv",
                    "pair_counts.csv",
                    "checkpoint_metrics.csv",
                    "final_summary.csv",
                }:
                    endpoint_rows.append(dict(row))
    except (OSError, UnicodeError, csv.Error) as exc:
        problems.append(f"{canonical}: cannot read CSV ({exc})")

    # Endpoint metadata is optional for legacy tables, though every value that
    # is present must agree with the measured checkpoint step.  This keeps a
    # 4T continuation row from silently entering a primary sample.
    primary_by_run: dict[str, int] = {}
    completed_by_run: dict[str, int] = {}
    for row_number, row in enumerate(endpoint_rows, start=2):
        run_id = str(row.get("run_id") or "")

        def parse_step(field: str) -> int | None:
            value = row.get(field)
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                try:
                    as_float = float(value)
                except (TypeError, ValueError):
                    problems.append(f"{canonical}: {field} is not integral at row {row_number}")
                    return None
                if not as_float.is_integer():
                    problems.append(f"{canonical}: {field} is not integral at row {row_number}")
                    return None
                parsed = int(as_float)
            if parsed < 0:
                problems.append(f"{canonical}: {field} is negative at row {row_number}")
                return None
            return parsed

        step = parse_step("step")
        primary = parse_step("primary_step")
        completed = parse_step("completed_step")
        if run_id and primary is not None:
            prior = primary_by_run.setdefault(run_id, primary)
            if prior != primary:
                problems.append(f"{canonical}: conflicting primary_step for run {run_id}")
        if run_id and completed is not None:
            prior = completed_by_run.setdefault(run_id, completed)
            if prior != completed:
                problems.append(f"{canonical}: conflicting completed_step for run {run_id}")
        if canonical in {"pair_counts.csv", "final_summary.csv"} and step is not None and primary is not None:
            if step != primary:
                problems.append(f"{canonical}: row step differs from primary_step at row {row_number}")
        if canonical == "checkpoint_metrics.csv":
            if (
                step is not None
                and primary is not None
                and str(row.get("is_primary") or "").strip().lower() in {"1", "true", "yes"}
                and step != primary
            ):
                problems.append(f"{canonical}: is_primary row differs from primary_step at row {row_number}")
            if (
                step is not None
                and completed is not None
                and str(row.get("is_terminal") or "").strip().lower() in {"1", "true", "yes"}
                and step != completed
            ):
                problems.append(f"{canonical}: is_terminal row differs from completed_step at row {row_number}")
        if primary is not None and completed is not None and primary > completed:
            problems.append(f"{canonical}: primary_step exceeds completed_step at row {row_number}")
    return problems


def _parse_checksums(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    problems: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return {}, [f"{CHECKSUMS_NAME}: cannot read ({exc})"]
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not _is_sha256(parts[0]):
            problems.append(f"{CHECKSUMS_NAME}: malformed line {number}")
            continue
        digest, filename = parts
        filename = filename.lstrip(" *")
        candidate = Path(filename)
        if candidate.is_absolute() or ".." in candidate.parts:
            problems.append(f"{CHECKSUMS_NAME}: unsafe path {filename}")
            continue
        if filename in values:
            problems.append(f"{CHECKSUMS_NAME}: duplicate path {filename}")
        values[filename] = digest.lower()
    return values, problems


def _manifest_path(root: Path) -> Path | None:
    candidates = [root / name for name in sorted(MANIFEST_NAMES) if (root / name).is_file()]
    return candidates[0] if candidates else None


def validate_compact_bundle(
    root: os.PathLike[str] | str,
    *,
    strict: bool = True,
    require_manifest: bool = True,
    require_checksums: bool = True,
    check_checksums: bool = True,
) -> list[str]:
    """Return all validation problems in a compact result bundle.

    An empty list means the bundle satisfies the contract.  The function is
    intentionally non-throwing so command-line verification can report every
    problem in one pass.  ``BundleValidationError`` is reserved for direct
    table-reading helpers.
    """

    bundle = Path(root)
    problems: list[str] = []
    if not bundle.is_dir():
        return [f"bundle directory does not exist: {bundle}"]

    files = _relative_files(bundle)
    manifest_files = [item.name for item in files if item.name in MANIFEST_NAMES]
    if len(manifest_files) > 1:
        problems.append("bundle: use exactly one manifest file")
    manifest = _manifest_path(bundle)
    if require_manifest and manifest is None:
        problems.append("bundle: missing bundle_manifest.json or manifest.json")
    if require_checksums and not (bundle / CHECKSUMS_NAME).is_file():
        problems.append(f"bundle: missing {CHECKSUMS_NAME}")
    csv_names = {
        item.as_posix()
        for item in files
        if len(item.parts) == 1 and item.suffix == ".csv"
    }
    if not csv_names:
        problems.append("bundle: no registered result tables")
    for required_name in ("stats.json", "provenance.json"):
        if not (bundle / required_name).is_file():
            problems.append(f"bundle: missing {required_name}")

    for relative in files:
        name = relative.as_posix()
        if relative.parts and len(relative.parts) != 1:
            problems.append(f"bundle: nested file is not allowed: {name}")
            continue
        if (bundle / relative).is_symlink():
            problems.append(f"bundle: symlink is not allowed: {name}")
            continue
        if name not in COMPACT_ALLOWLIST:
            if strict:
                problems.append(f"bundle: file is outside compact allowlist: {name}")
            continue
        canonical = canonical_name(name)
        if name.endswith(".csv"):
            problems.extend(_validate_table(bundle / relative, canonical, strict=strict))

    if manifest is not None:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                problems.append(f"{manifest.name}: top-level value must be an object")
            else:
                if payload.get("schema_version") != SCHEMA_VERSION:
                    problems.append(f"{manifest.name}: unsupported schema_version")
                if payload.get("status") != "complete":
                    problems.append(f"{manifest.name}: status must be complete")
                source_digest = payload.get("source_archive_sha256")
                if not isinstance(source_digest, str) or len(source_digest) != 64:
                    problems.append(
                        f"{manifest.name}: source_archive_sha256 must be a SHA-256 digest"
                    )
                table_entries = payload.get("tables")
                if not isinstance(table_entries, list) or not table_entries:
                    problems.append(f"{manifest.name}: tables must be a non-empty list")
                    table_entries = []
                declared_tables: set[str] = set()
                for entry in table_entries:
                    if not isinstance(entry, Mapping):
                        problems.append(f"{manifest.name}: table record must be an object")
                        continue
                    name = str(entry.get("path") or "")
                    digest = str(entry.get("sha256") or "").lower()
                    candidate = Path(name)
                    if not name or candidate.is_absolute() or ".." in candidate.parts:
                        problems.append(f"{manifest.name}: unsafe table path {name}")
                        continue
                    if name not in COMPACT_ALLOWLIST:
                        problems.append(f"{manifest.name}: table outside allowlist {name}")
                        continue
                    if not name.endswith(".csv"):
                        problems.append(f"{manifest.name}: table record is not CSV {name}")
                        continue
                    if name in declared_tables:
                        problems.append(f"{manifest.name}: duplicate table record {name}")
                    declared_tables.add(name)
                    table = bundle / candidate
                    if not table.is_file():
                        problems.append(f"{manifest.name}: table is missing {name}")
                    elif len(digest) != 64 or sha256_file(table) != digest:
                        problems.append(f"{manifest.name}: digest mismatch for {name}")
                for name in sorted(csv_names - declared_tables):
                    problems.append(f"{manifest.name}: unlisted table {name}")
                for name in sorted(declared_tables - csv_names):
                    problems.append(f"{manifest.name}: listed table is absent {name}")

                stats_record = payload.get("stats")
                if not isinstance(stats_record, Mapping):
                    problems.append(f"{manifest.name}: stats record is required")
                else:
                    stats_name = str(stats_record.get("path") or "")
                    stats_digest = str(stats_record.get("sha256") or "").lower()
                    stats_path = bundle / stats_name
                    if stats_name != "stats.json":
                        problems.append(f"{manifest.name}: stats path must be stats.json")
                    elif not stats_path.is_file():
                        problems.append(f"{manifest.name}: stats.json is missing")
                    elif len(stats_digest) != 64 or sha256_file(stats_path) != stats_digest:
                        problems.append(f"{manifest.name}: digest mismatch for stats.json")
                if strict:
                    provenance_record = payload.get("provenance")
                    provenance_path = bundle / "provenance.json"
                    if not isinstance(provenance_record, Mapping):
                        problems.append(f"{manifest.name}: provenance record is required")
                    else:
                        provenance_name = str(provenance_record.get("path") or "")
                        provenance_digest = str(provenance_record.get("sha256") or "").lower()
                        if provenance_name != "provenance.json":
                            problems.append(f"{manifest.name}: provenance path must be provenance.json")
                        elif not provenance_path.is_file():
                            problems.append(f"{manifest.name}: provenance.json is missing")
                        elif not _is_sha256(provenance_digest) or sha256_file(provenance_path) != provenance_digest:
                            problems.append(f"{manifest.name}: digest mismatch for provenance.json")
                    if provenance_path.is_file():
                        try:
                            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                        except (OSError, UnicodeError, json.JSONDecodeError):
                            provenance = None
                        if not isinstance(provenance, Mapping):
                            problems.append("provenance.json: top-level value must be an object")
                        else:
                            provenance_source = provenance.get("source_archive_sha256")
                            if not _is_sha256(provenance_source) or provenance_source != source_digest:
                                problems.append("provenance.json: source archive identity mismatch")
                            runtime = provenance.get("runtime")
                            if not isinstance(runtime, Mapping):
                                problems.append("provenance.json: runtime record is required")
                            else:
                                packages = runtime.get("packages")
                                accelerator = runtime.get("accelerator")
                                if not isinstance(packages, Mapping) or not packages:
                                    problems.append("provenance.json: dependency versions are required")
                                if not isinstance(accelerator, Mapping):
                                    problems.append("provenance.json: accelerator record is required")
                                else:
                                    available = accelerator.get(
                                        "available", accelerator.get("torch_cuda_available")
                                    )
                                    name = accelerator.get("name") or accelerator.get("torch_cuda_device")
                                    if available is not True or not str(name or "").strip():
                                        problems.append("provenance.json: live accelerator identity is required")
                    runs_path = bundle / "runs.csv"
                    if runs_path.is_file():
                        try:
                            run_rows = read_csv_rows(runs_path)
                        except (OSError, UnicodeError, csv.Error, BundleValidationError):
                            run_rows = []
                        run_ids = [str(row.get("run_id") or "").strip() for row in run_rows]
                        if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
                            problems.append("runs.csv: run IDs must be nonempty and unique")
                        digest_fields = (
                            "config_sha256", "train_dataset_sha256", "eval_dataset_sha256",
                            "reward_sha256", "objective_sha256", "source_archive_sha256",
                        )
                        for row_number, row in enumerate(run_rows, start=2):
                            for field in ("seed", "shuffle_seed", "git_commit"):
                                if not str(row.get(field) or "").strip():
                                    problems.append(f"runs.csv: missing {field} at row {row_number}")
                            if str(row.get("git_commit") or "").strip().lower() == "unknown":
                                problems.append(f"runs.csv: unknown git_commit at row {row_number}")
                            for field in digest_fields:
                                if not _is_sha256(row.get(field)):
                                    problems.append(f"runs.csv: invalid {field} at row {row_number}")
                            if row.get("source_archive_sha256") != source_digest:
                                problems.append(f"runs.csv: source archive mismatch at row {row_number}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            problems.append(f"{manifest.name}: invalid JSON ({exc})")

    checksums = bundle / CHECKSUMS_NAME
    if check_checksums and checksums.is_file():
        expected, checksum_problems = _parse_checksums(checksums)
        problems.extend(checksum_problems)
        actual_candidates = {
            item.as_posix()
            for item in files
            if item.as_posix() != CHECKSUMS_NAME and item.as_posix() in COMPACT_ALLOWLIST
        }
        if strict:
            missing = sorted(actual_candidates - expected.keys())
            extra = sorted(expected.keys() - actual_candidates)
            problems.extend(f"{CHECKSUMS_NAME}: missing entry {name}" for name in missing)
            problems.extend(f"{CHECKSUMS_NAME}: entry has no bundle file {name}" for name in extra)
        for name, expected_digest in expected.items():
            path = bundle / name
            if path.is_file() and sha256_file(path) != expected_digest:
                problems.append(f"{CHECKSUMS_NAME}: digest mismatch for {name}")

    return problems


def require_valid_bundle(root: os.PathLike[str] | str, **kwargs: object) -> Path:
    """Validate and raise one concise exception when the bundle is invalid."""

    problems = validate_compact_bundle(root, **kwargs)
    if problems:
        raise BundleValidationError("; ".join(problems))
    return Path(root)


def table_path(root: os.PathLike[str] | str, name: str) -> Path | None:
    """Find a canonical table, accepting documented legacy aliases."""

    bundle = Path(root)
    canonical = canonical_name(name)
    direct = bundle / canonical
    if direct.is_file():
        return direct
    for alias, target in TABLE_ALIASES.items():
        if target == canonical and (bundle / alias).is_file():
            return bundle / alias
    return None


def load_table(
    root: os.PathLike[str] | str,
    name: str,
    *,
    required: bool = False,
) -> list[dict[str, str]]:
    """Load one compact table, returning an empty list for an absent optional table."""

    path = table_path(root, name)
    if path is None:
        if required:
            raise BundleValidationError(f"missing table: {canonical_name(name)}")
        return []
    return read_csv_rows(path)


def manifest_payload(root: os.PathLike[str] | str) -> dict[str, object]:
    """Read the bundle manifest or return an empty payload for ad hoc analysis."""

    path = _manifest_path(Path(root))
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError("manifest must contain a JSON object")
    return payload


def file_hash_records(root: os.PathLike[str] | str) -> list[dict[str, str]]:
    """Return deterministic hashes for compact files other than the checksum sidecar."""

    bundle = Path(root)
    records: list[dict[str, str]] = []
    for relative in _relative_files(bundle):
        name = relative.as_posix()
        if name == CHECKSUMS_NAME or name not in COMPACT_ALLOWLIST:
            continue
        records.append({"path": name, "sha256": sha256_file(bundle / relative)})
    return records


def make_checksums(root: os.PathLike[str] | str) -> str:
    """Create deterministic sha256sum text for an already-written bundle."""

    records = file_hash_records(root)
    return "".join(f"{record['sha256']}  {record['path']}\n" for record in records)


def coerce_float(row: Mapping[str, object], field: str, default: float | None = None) -> float | None:
    """Parse a finite float from a CSV row."""

    value = row.get(field)
    if value in (None, ""):
        return default
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default
    return parsed


def coerce_int(row: Mapping[str, object], field: str, default: int | None = None) -> int | None:
    """Parse an integer from a CSV row, accepting integral float spellings."""

    value = row.get(field)
    if value in (None, ""):
        return default
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        try:
            as_float = float(str(value))
            if not as_float.is_integer():
                return default
            parsed = int(as_float)
        except (TypeError, ValueError):
            return default
    return parsed


__all__ = [
    "BundleValidationError",
    "CHECKSUMS_NAME",
    "COMPACT_ALLOWLIST",
    "KNOWN_COLUMNS",
    "MANIFEST_NAMES",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "canonical_name",
    "coerce_float",
    "coerce_int",
    "file_hash_records",
    "load_table",
    "make_checksums",
    "manifest_payload",
    "read_csv_rows",
    "require_valid_bundle",
    "sha256_file",
    "table_path",
    "validate_compact_bundle",
]
