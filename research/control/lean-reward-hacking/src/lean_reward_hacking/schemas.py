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

# The campaign writer emits these exact headers.  Keeping the required and
# known maps tied to the writer's contract makes strict validation reject
# accidental columns as well as raw-artifact files.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "runs.csv": (
        "schema_version", "experiment", "architecture", "condition", "phase", "level",
        "harm_strength", "audit_sensitivity", "harm_index", "audit_index", "seed_index",
        "cell_token", "run_id", "seed",
        "shuffle_seed", "config_sha256", "git_commit", "status", "final_step",
        "final_label", "threshold_set", "remote_raw_uri", "checkpoint_count",
    ),
    "pair_counts.csv": (
        "schema_version", "run_id", "branch_id", "source_run_id", "phase", "level",
        "harm_strength", "audit_sensitivity", "harm_index", "audit_index", "seed_index",
        "cell_token", "step", "eval_variant",
        "eval_set_hash", "n11", "n10", "n01", "n00", "n_pairs",
    ),
    "checkpoint_metrics.csv": (
        "schema_version", "run_id", "branch_id", "source_run_id", "phase", "level",
        "harm_strength", "audit_sensitivity", "harm_index", "audit_index", "seed_index",
        "cell_token", "step", "eval_variant",
        "eval_set_hash", "c_on", "c_off", "gap", "goal_score", "goal", "gate_effect",
        "gate", "off_audit_logit", "train_reward", "reward", "hidden_misalignment_rate",
        "hidden_misalignment", "train_loss", "n_pairs", "is_final", "label",
    ),
    "final_summary.csv": (
        "schema_version", "experiment", "architecture", "condition", "phase", "level",
        "harm_strength", "audit_sensitivity", "harm_index", "audit_index", "seed_index",
        "cell_token", "run_id", "seed", "step",
        "c_on", "c_off", "gap", "goal_score", "goal", "gate_effect", "gate", "off_audit_logit",
        "train_reward", "reward", "hidden_misalignment_rate", "hidden_misalignment", "label",
        "final_label", "threshold_set", "n_pairs", "eval_set_hash",
    ),
    "basin_cells.csv": (
        "schema_version", "experiment", "architecture", "harm_strength", "audit_sensitivity",
        "n_seeds", "n_complete", "n_invariant", "n_strategic", "n_intermediate", "p_invariant",
        "p_strategic", "p_intermediate", "ci_low", "ci_high", "config_sha256",
    ),
    "perturbation_trajectory.csv": (
        "schema_version", "source_run_id", "source_step", "source_label", "branch_id",
        "branch_kind", "control_kind", "intervention", "strength", "branch_seed", "step_since_branch",
        "c_on", "c_off", "gap", "goal_score", "goal", "gate_effect", "gate", "off_audit_logit",
        "reward", "hidden_misalignment_rate", "hidden_misalignment", "d_source", "d_opposite",
        "source_label_retention", "source_distance_closer", "intervention_feasible", "intervention_status",
        "mode_label", "recovery_status",
    ),
    "audit_control.csv": (
        "schema_version", "experiment", "architecture", "run_id", "control_kind", "available",
        "reason", "baseline_c_on", "baseline_c_off", "baseline_gap", "control_c_on", "control_c_off",
        "control_gap", "delta_c_on", "delta_c_off", "delta_gap", "n_pairs", "eval_set_hash",
    ),
    "threshold_sensitivity.csv": (
        "schema_version", "experiment", "architecture", "threshold_set", "c_on_min",
        "invariant_c_off_min", "strategic_c_off_max", "require_goal_sign", "n_runs", "n_invariant",
        "n_strategic", "n_intermediate", "labels_json", "counts_json", "status",
    ),
}

KNOWN_COLUMNS: dict[str, frozenset[str]] = {
    name: frozenset(fields) for name, fields in REQUIRED_COLUMNS.items()
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
    except (OSError, UnicodeError, csv.Error) as exc:
        problems.append(f"{canonical}: cannot read CSV ({exc})")
    return problems


def _validate_cross_table_rows(bundle: Path) -> list[str]:
    """Catch stale trajectory rows that satisfy the CSV header contract."""

    problems: list[str] = []
    runs_path = bundle / "runs.csv"
    if not runs_path.is_file():
        return problems
    try:
        runs = read_csv_rows(runs_path)
    except BundleValidationError:
        return problems
    final_steps: dict[str, int] = {}
    for row_number, row in enumerate(runs, start=2):
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        value = row.get("final_step")
        if value in (None, ""):
            continue
        try:
            step = int(str(value))
        except (TypeError, ValueError):
            problems.append(f"runs.csv: invalid final_step at row {row_number}")
            continue
        if step < 1:
            problems.append(f"runs.csv: final_step must be positive at row {row_number}")
        if run_id in final_steps:
            problems.append(f"runs.csv: duplicate run_id {run_id}")
        final_steps[run_id] = step

    metrics_path = bundle / "checkpoint_metrics.csv"
    if metrics_path.is_file():
        try:
            rows = read_csv_rows(metrics_path)
        except BundleValidationError:
            rows = []
        seen: set[tuple[str, int, str]] = set()
        observed: dict[str, set[int]] = {run_id: set() for run_id in final_steps}
        for row_number, row in enumerate(rows, start=2):
            run_id = str(row.get("run_id") or "")
            if run_id not in final_steps:
                continue
            try:
                step = int(str(row.get("step") or ""))
            except (TypeError, ValueError):
                problems.append(f"checkpoint_metrics.csv: invalid step at row {row_number}")
                continue
            if step < 1 or step > final_steps[run_id]:
                problems.append(f"checkpoint_metrics.csv: stale step for {run_id} at row {row_number}")
            branch_id = str(row.get("branch_id") or "")
            key = (run_id, step, branch_id)
            if key in seen:
                problems.append(f"checkpoint_metrics.csv: duplicate row for {run_id} at step {step}")
            seen.add(key)
            observed[run_id].add(step)
            if step == final_steps[run_id] and str(row.get("is_final") or "").lower() not in {"true", "1"}:
                problems.append(f"checkpoint_metrics.csv: target row for {run_id} is not final")
        for run_id, final_step in final_steps.items():
            if observed.get(run_id) and final_step not in observed[run_id]:
                problems.append(f"checkpoint_metrics.csv: missing final row for {run_id}")

    final_path = bundle / "final_summary.csv"
    if final_path.is_file():
        try:
            rows = read_csv_rows(final_path)
        except BundleValidationError:
            rows = []
        seen: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            run_id = str(row.get("run_id") or "")
            if run_id not in final_steps:
                continue
            if run_id in seen:
                problems.append(f"final_summary.csv: duplicate run_id {run_id}")
            seen.add(run_id)
            try:
                step = int(str(row.get("step") or ""))
            except (TypeError, ValueError):
                problems.append(f"final_summary.csv: invalid step at row {row_number}")
                continue
            if step != final_steps[run_id]:
                problems.append(f"final_summary.csv: stale final step for {run_id}")
        if rows and final_steps and set(final_steps).issubset(seen) is False:
            # A legacy bundle may omit final_summary entirely, but once the
            # table is present it must contain each declared completed run.
            problems.extend(
                f"final_summary.csv: missing final row for {run_id}"
                for run_id in sorted(set(final_steps) - seen)
            )
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
        if len(parts) != 2 or len(parts[0]) != 64:
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

    problems.extend(_validate_cross_table_rows(bundle))

    if manifest is not None:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                problems.append(f"{manifest.name}: top-level value must be an object")
            elif payload.get("schema_version") != SCHEMA_VERSION:
                problems.append(f"{manifest.name}: unsupported schema_version")
            elif isinstance(payload.get("tables"), list):
                for entry in payload["tables"]:
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
                    table = bundle / candidate
                    if not table.is_file():
                        problems.append(f"{manifest.name}: table is missing {name}")
                    elif len(digest) != 64 or sha256_file(table) != digest:
                        problems.append(f"{manifest.name}: digest mismatch for {name}")
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
