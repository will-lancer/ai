"""Claim-gated report rendering from compact experiment evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .analysis import analyze_bundle, load_final_rows
from .schemas import file_hash_records, load_table, manifest_payload


CLAIM_NAMES = {
    1: "different final hidden behaviors",
    2: "two final statistical modes",
    3: "modes survive longer training",
    4: "source-conditioned attraction",
    5: "generic-network and language-model support",
}


# These values are registered in configs/perturbation_colab.toml.  Keep the
# report gate independent from the campaign implementation so a status marker
# cannot silently weaken the evidence requirement.
REGISTERED_MINIMUM_RECOVERY_FRACTION = 0.50
REGISTERED_MINIMUM_SOURCE_RETENTION = 0.80
REGISTERED_MINIMUM_INTERVENTION_FAMILIES = 2
REGISTERED_MINIMUM_BIC_DELTA = 10.0
REGISTERED_MAXIMUM_MIXTURE_P_VALUE = 0.05
REGISTERED_MINIMUM_COMPONENT_WEIGHT = 0.10
REGISTERED_MINIMUM_COMPONENT_SEPARATION = 0.30
_SOURCE_LABELS = frozenset({"oversight-invariant", "strategic"})
_CONTROL_BRANCH_KINDS = frozenset({"sham", "frozen"})
_RESET_BRANCH_KINDS = frozenset({"reset_optimizer", "reset-optimizer", "reset optimizer"})
_INTERVENTION_BRANCH_KINDS = frozenset({"resumed", "intervention", "perturbation", "dynamic"})
_RECOVERY_STATUS_VALUES = frozenset({"recovered", "complete", "completed", "ok"})
_FLOAT_TOLERANCE = 1e-9


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "complete", "ok", "recovered"}


def _mapping_value(mapping: Mapping[str, object], *fields: str) -> object | None:
    """Return the first present field, preserving an explicitly missing value."""

    for field in fields:
        if field in mapping and mapping.get(field) not in (None, ""):
            return mapping.get(field)
    return None


def _normalised_token(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _metric_token(value: object) -> str:
    token = _normalised_token(value).replace("(", "").replace(")", "")
    if token in {"logitc-off", "logit-coff", "logitcoff"}:
        return "logit-c-off"
    return token


def _primary_mixture_support(stats: Mapping[str, object]) -> tuple[bool, str]:
    """Apply the registered modality gate to the logit(C_off) mixture only.

    The p-value is accepted only from the mixture result's fitted-null
    parametric bootstrap.  Gap-scale dip and Silverman results are reported as
    secondary diagnostics by the caller and never enter this decision.
    """

    bimodality = stats.get("bimodality")
    if not isinstance(bimodality, Mapping):
        return False, "primary mixture failed: bimodality summary is absent"

    primary = bimodality.get("primary")
    mixture = bimodality.get("mixture")
    if not isinstance(primary, Mapping) or not isinstance(mixture, Mapping):
        return False, "primary mixture failed: primary and mixture summaries are required"

    primary_scale = _metric_token(
        _mapping_value(bimodality, "primary_scale")
        or _mapping_value(primary, "scale", "metric")
    )
    primary_metric = _metric_token(
        _mapping_value(bimodality, "primary_metric")
        or _mapping_value(primary, "metric", "scale")
    )
    primary_method = _normalised_token(
        _mapping_value(bimodality, "primary_method")
        or _mapping_value(primary, "method")
    )
    if primary_scale != "logit-c-off" or primary_metric != "logit-c-off":
        return False, (
            "primary mixture failed: registered scale/metric must be logit_c_off "
            f"(scale={primary_scale or 'missing'}; metric={primary_metric or 'missing'})"
        )
    if primary_method not in {"gaussian-mixture-bic", "gaussian-mixture-bic-bootstrap"}:
        return False, (
            "primary mixture failed: registered method must be gaussian_mixture_bic "
            f"(method={primary_method or 'missing'})"
        )

    fit = _mapping_value(primary, "fit")
    if not isinstance(fit, Mapping):
        fit = _mapping_value(mixture, "fit")
    if not isinstance(fit, Mapping):
        return False, "primary mixture failed: fitted two-component result is absent"

    bic_delta = _finite_float(
        _mapping_value(
            primary,
            "bic_delta",
            "delta_bic",
            "observed_bic_delta",
        )
        if _mapping_value(primary, "bic_delta", "delta_bic", "observed_bic_delta") is not None
        else _mapping_value(
            fit,
            "bic_delta",
            "delta_bic",
            "observed_bic_delta",
        )
    )
    if bic_delta is None:
        bic_delta = _finite_float(
            _mapping_value(mixture, "observed_bic_delta", "bic_delta", "delta_bic")
        )

    p_value = _finite_float(
        _mapping_value(
            mixture,
            "bic_delta_p_value",
            "delta_bic_p_value",
            "p_value",
        )
    )

    calibration = _normalised_token(
        _mapping_value(mixture, "calibration", "bootstrap_calibration")
    )
    null_model = _normalised_token(
        _mapping_value(mixture, "null_model")
    )
    statistic = _normalised_token(
        _mapping_value(mixture, "statistic")
    )
    calibration_ok = (
        "parametric-bootstrap" in calibration
        and "fitted-one-component-gaussian-null" in calibration
        and null_model == "one-component-gaussian-mle"
        and statistic == "bic-delta"
    )

    weights_value = _mapping_value(fit, "weights", "component_weights")
    weights: list[float] = []
    if isinstance(weights_value, Sequence) and not isinstance(weights_value, (str, bytes)):
        parsed_weights = [_finite_float(value) for value in weights_value]
        if (
            len(parsed_weights) == 2
            and all(value is not None for value in parsed_weights)
            and all(0.0 <= float(value) <= 1.0 for value in parsed_weights if value is not None)
            and abs(sum(float(value) for value in parsed_weights if value is not None) - 1.0)
            <= 1e-6
        ):
            weights = [float(value) for value in parsed_weights if value is not None]
    minimum_weight = min(weights) if weights else None

    separation = _finite_float(
        _mapping_value(
            fit,
            "separation",
            "component_separation",
            "observed_separation",
        )
    )
    if separation is None:
        separation = _finite_float(
            _mapping_value(primary, "separation", "component_separation")
            or _mapping_value(mixture, "separation", "component_separation")
        )

    separation_threshold = REGISTERED_MINIMUM_COMPONENT_SEPARATION
    configured_summaries = [primary, mixture]
    for candidate in (bimodality.get("configuration"), stats.get("configuration")):
        if isinstance(candidate, Mapping):
            configured_summaries.append(candidate)
    for summary in configured_summaries:
        configured = _finite_float(
            _mapping_value(
                summary,
                "minimum_separation",
                "minimum_component_separation",
                "minimum_gap_separation",
            )
        )
        if configured is not None:
            separation_threshold = max(separation_threshold, configured)

    bic_ok = bic_delta is not None and bic_delta >= REGISTERED_MINIMUM_BIC_DELTA
    p_ok = (
        p_value is not None
        and 0.0 <= p_value <= REGISTERED_MAXIMUM_MIXTURE_P_VALUE
        and calibration_ok
    )
    weight_ok = minimum_weight is not None and minimum_weight >= REGISTERED_MINIMUM_COMPONENT_WEIGHT
    separation_ok = (
        separation is not None and separation >= separation_threshold
    )

    secondary: list[str] = []
    for key in ("dip", "silverman"):
        result = bimodality.get(key)
        if not isinstance(result, Mapping):
            continue
        status = _normalised_token(result.get("status"))
        p_secondary = _finite_float(result.get("p_value"))
        if status or p_secondary is not None:
            secondary.append(
                f"{key}="
                f"{status or 'available'}"
                f"{'' if p_secondary is None else f' p={p_secondary:.4g}'}"
            )
    detail = (
        f"primary logit(C_off) mixture; "
        f"delta_bic={('missing' if bic_delta is None else f'{bic_delta:.4g}')}>={REGISTERED_MINIMUM_BIC_DELTA:g}; "
        f"bootstrap_p={('missing' if p_value is None else f'{p_value:.4g}')}<={REGISTERED_MAXIMUM_MIXTURE_P_VALUE:g}; "
        f"fitted_null_calibration={calibration_ok}; "
        f"minimum_component_weight={('missing' if minimum_weight is None else f'{minimum_weight:.4g}')}>={REGISTERED_MINIMUM_COMPONENT_WEIGHT:g}; "
        f"separation={('missing' if separation is None else f'{separation:.4g}')}>={separation_threshold:g}"
    )
    if secondary:
        detail += "; secondary=" + ", ".join(secondary)
    supported = bool(bic_ok and p_ok and weight_ok and separation_ok)
    if not supported:
        detail += "; primary criterion failed"
    return supported, detail


def _dip_or_silverman_support(stats: Mapping[str, object], alpha: float) -> tuple[bool, str]:
    """Compatibility name for the primary gate.

    Older callers used this private helper for a generic modality decision.
    The report gate now delegates to the registered logit-mixture criteria.
    """

    del alpha
    return _primary_mixture_support(stats)


def _label_counts(finals: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("label", "intermediate")) for row in finals)


def _extended_training_support(trajectory: Sequence[Mapping[str, object]], finals: Sequence[Mapping[str, object]]) -> tuple[bool, str]:
    if not trajectory:
        return False, "checkpoint trajectories are absent"
    final_steps: dict[str, float] = {}
    labels: dict[str, set[str]] = {}
    for row in trajectory:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        try:
            step = float(row.get("step", ""))
        except (TypeError, ValueError):
            continue
        final_steps[run_id] = max(step, final_steps.get(run_id, float("-inf")))
        label = str(row.get("label") or row.get("final_label") or "").strip()
        if label:
            labels.setdefault(run_id, set()).add(label)
    long_runs = sum(1 for step in final_steps.values() if step > 4000)
    endpoint_labels = set()
    for row in finals:
        label = str(row.get("label", ""))
        if label in {"oversight-invariant", "strategic"}:
            endpoint_labels.add(label)
    supported = long_runs > 0 and len(endpoint_labels) >= 2
    return supported, f"long_runs={long_runs}; endpoint_labels={sorted(endpoint_labels)}"


def _finite_float(value: object) -> float | None:
    """Parse a finite scalar without treating booleans as measurements."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_float(row: Mapping[str, object], *fields: str) -> float | None:
    for field in fields:
        if field in row and row.get(field) not in (None, ""):
            parsed = _finite_float(row.get(field))
            if parsed is not None:
                return parsed
    return None


def _row_text(row: Mapping[str, object], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            text = str(value).strip().lower()
            if text:
                return text
    return ""


def _canonical_source_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if text in _SOURCE_LABELS:
        return text
    return ""


def _source_label(row: Mapping[str, object]) -> str:
    for field in ("source_label", "source_mode", "source_endpoint_label", "source_class"):
        label = _canonical_source_label(row.get(field))
        if label:
            return label
    return ""


def _branch_label(row: Mapping[str, object]) -> str:
    """Return the observed branch endpoint label, excluding source metadata."""

    for field in (
        "mode_label",
        "branch_label",
        "final_mode",
        "final_label",
        "endpoint_label",
        "endpoint_mode",
        "observed_label",
        "recovered_label",
        "label",
    ):
        value = row.get(field)
        if value not in (None, ""):
            text = str(value).strip().lower().replace("_", "-")
            if text:
                return text
    return ""


def _intervention_family(row: Mapping[str, object]) -> str:
    """Resolve a family while collapsing intervention strengths into one family."""

    explicit = row.get("intervention_family")
    if explicit not in (None, ""):
        return str(explicit).strip().lower().replace(" ", "_")
    for field in ("intervention", "family", "intervention_kind", "kind"):
        value = row.get(field)
        if value in (None, ""):
            continue
        text = str(value).strip().lower().replace(" ", "_")
        # Strengths and run-specific suffixes do not constitute new families.
        for separator in (":", "|", "@", "#"):
            text = text.split(separator, 1)[0]
        if text:
            return text
    return ""


def _branch_kind(row: Mapping[str, object]) -> str:
    """Return the registered branch role, keeping unknown roles explicit."""

    for field in (
        "branch_kind",
        "control_kind",
        "branch_type",
        "control_type",
        "endpoint_type",
    ):
        value = row.get(field)
        if value in (None, ""):
            continue
        token = _normalised_token(value)
        if token:
            return token
    value = row.get("kind")
    if value not in (None, ""):
        token = _normalised_token(value)
        if token in _CONTROL_BRANCH_KINDS | _RESET_BRANCH_KINDS | _INTERVENTION_BRANCH_KINDS:
            return token
    return ""


def _optimizer_policy(row: Mapping[str, object]) -> str:
    """Resolve optimizer provenance without trusting a status marker."""

    for field in ("optimizer_policy", "optimizer", "optimizer_state_policy", "optimizer_state"):
        value = row.get(field)
        if value in (None, ""):
            continue
        token = _normalised_token(value)
        if token in {"preserve", "preserved", "same", "retained"}:
            return "preserve"
        if token in {"reset", "reset-optimizer", "reset-moments", "fresh"}:
            return "reset"
        return token
    for field in ("optimizer_reset", "reset_optimizer", "optimizer_state_reset"):
        if field in row and row.get(field) not in (None, ""):
            value = row.get(field)
            if isinstance(value, bool):
                return "reset" if value else "preserve"
            token = _normalised_token(value)
            if token in {"true", "yes", "1", "reset", "reset-optimizer"}:
                return "reset"
            if token in {"false", "no", "0", "preserve", "preserved"}:
                return "preserve"
            return token
    return ""


def _source_run_id(row: Mapping[str, object]) -> str:
    for field in ("source_run_id", "source_id", "source_run", "parent_source_run_id"):
        value = row.get(field)
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return ""


def _source_step(row: Mapping[str, object]) -> float | None:
    return _row_float(row, "source_step", "source_checkpoint", "checkpoint_step")


def _explicit_match_tokens(row: Mapping[str, object]) -> set[str]:
    tokens: set[str] = set()
    for field in (
        "control_group",
        "matched_group",
        "match_id",
        "lineage_id",
        "lineage_group",
        "intervention_group",
        "source_branch_id",
        "parent_branch_id",
        "matched_branch_id",
        "resumed_branch_id",
    ):
        value = row.get(field)
        if value not in (None, ""):
            token = str(value).strip()
            if token:
                tokens.add(token)
    return tokens


def _control_family(row: Mapping[str, object], role: str) -> str:
    explicit = row.get("intervention_family")
    if explicit not in (None, ""):
        return str(explicit).strip().lower().replace(" ", "_")
    family = _intervention_family(row)
    if role == "sham" and family in {"identity", "sham"}:
        return ""
    return family


def _branch_has_measured_rows(rows: Sequence[Mapping[str, object]], *, require_opposite: bool) -> bool:
    """Require actual trajectory measurements for every branch role."""

    if not rows:
        return False
    for row in rows:
        if _row_float(row, "step_since_branch", "horizon", "step", "checkpoint") is None:
            return False
        if _row_float(row, "d_source", "source_distance") is None:
            return False
        if require_opposite and _row_float(row, "d_opposite", "opposite_distance") is None:
            return False
    return True


def _control_matches(
    control: Mapping[str, object],
    intervention: Mapping[str, object],
    *,
    role: str,
) -> bool:
    """Match a sham/frozen branch to one intervention source branch."""

    control_source = _source_run_id(control)
    intervention_source = _source_run_id(intervention)
    control_tokens = _explicit_match_tokens(control)
    intervention_tokens = _explicit_match_tokens(intervention)
    intervention_branch = str(intervention.get("branch_id") or "").strip()
    control_branch = str(control.get("branch_id") or "").strip()

    if control_source and intervention_source and control_source != intervention_source:
        return False
    if not control_source and not intervention_source:
        if not control_tokens or not intervention_tokens:
            return False
    elif not control_source or not intervention_source:
        return False

    control_step = _source_step(control)
    intervention_step = _source_step(intervention)
    if control_step is not None and intervention_step is not None:
        if abs(control_step - intervention_step) > _FLOAT_TOLERANCE:
            return False
    elif (control_step is not None or intervention_step is not None) and not control_tokens.intersection(
        intervention_tokens
    ):
        return False

    explicit_match = bool(control_tokens.intersection(intervention_tokens))
    if intervention_branch and intervention_branch in control_tokens:
        explicit_match = True
    if control_branch and control_branch in intervention_tokens:
        explicit_match = True
    if control_tokens and intervention_tokens and not explicit_match:
        return False

    intervention_family = _intervention_family(intervention)
    control_family = _control_family(control, role)
    if control_family and intervention_family and control_family != intervention_family:
        return False
    return True


def _branch_key(row: Mapping[str, object], index: int) -> str:
    for field in ("branch_id", "branch_run_id", "branch", "branch_name", "run_id"):
        value = row.get(field)
        if value not in (None, ""):
            return f"{field}:{str(value).strip()}"
    source = str(row.get("source_run_id") or "").strip()
    family = _intervention_family(row)
    strength = str(row.get("strength") or "").strip()
    if source and family:
        return f"source:{source}|family:{family}|strength:{strength}"
    return f"row:{index}"


def _ordered_branch_rows(rows: Sequence[tuple[int, Mapping[str, object]]]) -> list[Mapping[str, object]]:
    def sort_key(item: tuple[int, Mapping[str, object]]) -> tuple[float, float, int]:
        index, row = item
        step = _row_float(row, "step_since_branch", "horizon", "step", "checkpoint")
        return (float("inf") if step is None else step, float(index), index)

    return [row for _, row in sorted(rows, key=sort_key)]


def _branch_is_frozen(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        kinds = {
            _row_text(row, field)
            for field in ("branch_kind", "control_kind", "endpoint_type", "endpoint_classification")
        }
        status = _row_text(row, "recovery_status")
        if kinds.intersection(_CONTROL_BRANCH_KINDS | {"frozen_endpoint", "flat", "flat_direction"}):
            return True
        if status in {"frozen", "frozen_endpoint", "flat", "flat_direction"}:
            return True
        if _truth(row.get("frozen")) or _truth(row.get("frozen_endpoint")):
            return True
    return False


def _branch_is_shared(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        kinds = {
            _row_text(row, field)
            for field in ("endpoint_type", "endpoint_classification", "recovery_status")
        }
        if kinds.intersection({"shared", "shared_endpoint", "shared-endpoint"}):
            return True
        if _truth(row.get("shared_endpoint")):
            return True
    return False


def _branch_is_persistent_intermediate(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        statuses = {
            _row_text(row, field)
            for field in ("recovery_status", "endpoint_type", "endpoint_classification")
        }
        if statuses.intersection({"persistent_intermediate", "persistent-intermediate"}):
            return True
        if _truth(row.get("persistent_intermediate")):
            return True
    return False


def _recovery_measurement(rows: Sequence[Mapping[str, object]]) -> tuple[float | None, bool, str]:
    """Return final recovery fraction, monotonicity, and evidence source."""

    fractions = [
        value
        for row in rows
        for value in (_row_float(row, "recovery_fraction"),)
        if value is not None
    ]
    final_fraction = _row_float(
        rows[-1],
        "recovery_fraction_final",
        "final_recovery_fraction",
        "registered_recovery_fraction",
        "recovered_fraction",
    )
    if final_fraction is None and fractions:
        final_fraction = fractions[-1]

    distances = [
        value
        for row in rows
        for value in (_row_float(row, "d_source", "source_distance"),)
        if value is not None
    ]
    if final_fraction is None:
        initial_distance = _row_float(
            rows[0],
            "source_distance_initial",
            "initial_source_distance",
            "d_source_initial",
        )
        final_distance = _row_float(
            rows[-1],
            "source_distance_final",
            "final_source_distance",
            "d_source_final",
        )
        if initial_distance is not None and final_distance is not None and initial_distance > 0.0:
            final_fraction = (initial_distance - final_distance) / initial_distance
        elif len(distances) >= 2 and distances[0] > 0.0:
            final_fraction = (distances[0] - distances[-1]) / distances[0]

    monotonic_marker: bool | None = None
    for row in rows:
        for field in (
            "monotonic_source_recovery",
            "monotonic_recovery",
            "source_directed_monotonic",
            "monotonic",
        ):
            if field in row and row.get(field) not in (None, ""):
                monotonic_marker = _truth(row.get(field))
                break
        if monotonic_marker is not None:
            break

    monotonic = True
    sequence = fractions if len(fractions) >= 2 else distances
    if len(sequence) >= 2:
        if fractions:
            monotonic = all(
                sequence[index] + _FLOAT_TOLERANCE >= sequence[index - 1]
                for index in range(1, len(sequence))
            )
        else:
            monotonic = all(
                sequence[index] <= sequence[index - 1] + _FLOAT_TOLERANCE
                for index in range(1, len(sequence))
            )
    elif monotonic_marker is not None:
        monotonic = monotonic_marker
    else:
        # A scalar fraction without a trajectory cannot establish direction.
        monotonic = False
    if monotonic_marker is False:
        monotonic = False
    evidence = "trajectory" if len(sequence) >= 2 else ("registered scalar" if final_fraction is not None else "absent")
    return final_fraction, monotonic, evidence


def _source_retention(rows: Sequence[Mapping[str, object]], source_label: str) -> tuple[float | None, str]:
    scalar_fields = (
        "source_label_retention",
        "source_label_retention_fraction",
        "source_retention",
        "late_source_persistence",
        "late_source_retention",
        "source_label_persistence",
        "retention",
    )
    for row in reversed(rows):
        value = _row_float(row, *scalar_fields)
        if value is not None:
            return value, "registered scalar"

    labels = [_branch_label(row) for row in rows]
    labels = [label for label in labels if label]
    if not labels:
        return None, "absent"
    late_count = max(1, math.ceil(len(labels) * 0.20))
    late_labels = labels[-late_count:]
    return sum(label == source_label for label in late_labels) / len(late_labels), "late branch labels"


def _attraction_support(
    perturbations: Sequence[Mapping[str, object]],
    *,
    minimum_recovery_fraction: float = REGISTERED_MINIMUM_RECOVERY_FRACTION,
    minimum_source_retention: float = REGISTERED_MINIMUM_SOURCE_RETENTION,
    minimum_intervention_families: int = REGISTERED_MINIMUM_INTERVENTION_FAMILIES,
) -> tuple[bool, str]:
    """Require measured, source-directed recovery under preserved optimizer state."""

    if not 0.0 <= minimum_recovery_fraction <= 1.0:
        raise ValueError("minimum_recovery_fraction must lie in [0, 1]")
    if not 0.0 <= minimum_source_retention <= 1.0:
        raise ValueError("minimum_source_retention must lie in [0, 1]")
    if (
        isinstance(minimum_intervention_families, bool)
        or not isinstance(minimum_intervention_families, int)
        or minimum_intervention_families < 1
    ):
        raise ValueError("minimum_intervention_families must be a positive integer")
    if not perturbations:
        return False, "perturbation trajectories are absent"
    grouped: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for index, row in enumerate(perturbations):
        grouped.setdefault(_branch_key(row, index), []).append((index, row))

    rows_by_key = {
        key: _ordered_branch_rows(indexed_rows)
        for key, indexed_rows in grouped.items()
    }
    branch_evidence: list[dict[str, object]] = []
    for key, rows in rows_by_key.items():
        roles = {_branch_kind(row) for row in rows}
        role = next(iter(roles), "") if len(roles) == 1 else ""
        if role in _RESET_BRANCH_KINDS:
            role = "reset_optimizer"
        elif role in _CONTROL_BRANCH_KINDS:
            role = role.replace("-", "_")
        elif role in _INTERVENTION_BRANCH_KINDS:
            role = "intervention"
        else:
            role = "unknown"
        if role == "intervention" and rows and all(
            _optimizer_policy(row) == "reset" for row in rows
        ):
            role = "reset_optimizer"

        source_labels = [_source_label(row) for row in rows]
        source_label = (
            source_labels[0]
            if source_labels and source_labels[0] in _SOURCE_LABELS and all(
                label == source_labels[0] for label in source_labels
            )
            else ""
        )
        family_values = {_intervention_family(row) for row in rows if _intervention_family(row)}
        family = next(iter(family_values), "") if len(family_values) == 1 else ""
        source_run_id = _source_run_id(rows[0]) if rows else ""
        measured = _branch_has_measured_rows(rows, require_opposite=role == "intervention")
        preserve_optimizer = bool(rows) and all(
            _optimizer_policy(row) == "preserve" for row in rows
        )
        frozen = role == "frozen"
        sham = role == "sham"
        reset_optimizer = role == "reset_optimizer"
        shared = _branch_is_shared(rows)
        persistent_intermediate = _branch_is_persistent_intermediate(rows)
        fraction, monotonic, fraction_source = (
            _recovery_measurement(rows) if role == "intervention" and measured else (None, False, "absent")
        )
        retention, retention_source = (
            _source_retention(rows, source_label)
            if role == "intervention" and source_label and measured
            else (None, "absent")
        )
        statuses = {
            _row_text(row, "recovery_status")
            for row in rows
            if _row_text(row, "recovery_status")
        }
        terminal_statuses = statuses.difference({"pending", "in_progress", "running"})
        status_ok = not terminal_statuses or terminal_statuses.issubset(_RECOVERY_STATUS_VALUES)
        endpoint = _branch_label(rows[-1]) if rows else ""
        fraction_ok = (
            fraction is not None
            and 0.0 <= fraction <= 1.0 + _FLOAT_TOLERANCE
            and fraction >= minimum_recovery_fraction
        )
        retention_ok = retention is not None and retention >= minimum_source_retention

        opposite_distances = [
            _row_float(row, "d_opposite", "opposite_distance") for row in rows
        ]
        source_distances = [_row_float(row, "d_source", "source_distance") for row in rows]
        source_directed = False
        opposite_endpoint_rejected = False
        if (
            role == "intervention"
            and measured
            and len(rows) >= 2
            and all(value is not None for value in (*source_distances, *opposite_distances))
        ):
            source_final = float(source_distances[-1])  # type: ignore[arg-type]
            opposite_initial = float(opposite_distances[0])  # type: ignore[arg-type]
            opposite_final = float(opposite_distances[-1])  # type: ignore[arg-type]
            source_directed = (
                source_final + _FLOAT_TOLERANCE < opposite_final
                and opposite_final + _FLOAT_TOLERANCE >= opposite_initial
            )
            opposite_endpoint_rejected = not source_directed
            for row in rows:
                for field in (
                    "source_directed",
                    "source_directed_recovery",
                    "source_directed_monotonic",
                ):
                    if field in row and row.get(field) not in (None, "") and not _truth(row.get(field)):
                        source_directed = False
                        opposite_endpoint_rejected = True

        qualifies = bool(
            role == "intervention"
            and source_label
            and source_run_id
            and family
            and measured
            and preserve_optimizer
            and not shared
            and not persistent_intermediate
            and status_ok
            and fraction_source == "trajectory"
            and fraction_ok
            and monotonic
            and source_directed
            and retention_ok
        )
        branch_evidence.append(
            {
                "key": key,
                "role": role,
                "source_label": source_label,
                "source_run_id": source_run_id,
                "family": family,
                "measured": measured,
                "preserve_optimizer": preserve_optimizer,
                "frozen": frozen,
                "sham": sham,
                "reset_optimizer": reset_optimizer,
                "shared": shared,
                "persistent_intermediate": persistent_intermediate,
                "status_ok": status_ok,
                "fraction": fraction,
                "fraction_ok": fraction_ok,
                "fraction_source": fraction_source,
                "monotonic": monotonic,
                "retention": retention,
                "retention_ok": retention_ok,
                "retention_source": retention_source,
                "source_directed": source_directed,
                "opposite_endpoint_rejected": opposite_endpoint_rejected,
                "endpoint": endpoint,
                "qualifies": qualifies,
            }
        )

    intervention_evidence = [
        item for item in branch_evidence if item["role"] == "intervention"
    ]
    controls = [item for item in branch_evidence if item["role"] in _CONTROL_BRANCH_KINDS]
    reset_controls = [item for item in branch_evidence if item["role"] == "reset_optimizer"]
    invalid = [item for item in branch_evidence if item["role"] == "unknown"]

    # Controls are matched by source checkpoint and, where supplied, lineage
    # tokens.  A sham uses identity as its intervention name, so its explicit
    # family is optional while the source match remains mandatory.
    for item in intervention_evidence:
        dynamic_rows = rows_by_key.get(str(item["key"]), [])
        if not dynamic_rows:
            item["matched_controls"] = {"sham": False, "frozen": False}
            item["controls_ok"] = False
            continue
        dynamic = dynamic_rows[0]
        matches: dict[str, bool] = {}
        for role_name in _CONTROL_BRANCH_KINDS:
            matches[role_name] = any(
                bool(control["measured"])
                and (
                    not str(control["source_label"])
                    or str(control["source_label"]) == str(item["source_label"])
                )
                and _control_matches(
                    rows_by_key.get(str(control["key"]), [{}])[0],
                    dynamic,
                    role=role_name,
                )
                for control in controls
                if control["role"] == role_name
            )
        item["matched_controls"] = matches
        item["controls_ok"] = bool(matches.get("sham") and matches.get("frozen"))
        item["qualifies"] = bool(item["qualifies"] and item["controls_ok"])

    frozen_count = sum(bool(item["frozen"]) for item in branch_evidence)
    sham_count = sum(bool(item["sham"]) for item in branch_evidence)
    reset_count = sum(bool(item["reset_optimizer"]) for item in branch_evidence)
    persistent_intermediate_count = sum(bool(item["persistent_intermediate"]) for item in branch_evidence)
    explicit_shared = sum(bool(item["shared"]) for item in intervention_evidence)
    source_classes = set()
    families_by_source: dict[str, set[str]] = {label: set() for label in _SOURCE_LABELS}
    qualifying = [item for item in branch_evidence if item["qualifies"]]
    for item in qualifying:
        source = str(item["source_label"])
        source_classes.add(source)
        families_by_source.setdefault(source, set()).add(str(item["family"]))

    endpoints_by_source: dict[str, set[str]] = {label: set() for label in _SOURCE_LABELS}
    for item in intervention_evidence:
        source = str(item["source_label"])
        endpoint = str(item["endpoint"])
        if source in _SOURCE_LABELS and endpoint:
            endpoints_by_source.setdefault(source, set()).add(endpoint)
    endpoint_sources: dict[str, set[str]] = {}
    for source, endpoints in endpoints_by_source.items():
        for endpoint in endpoints:
            endpoint_sources.setdefault(endpoint, set()).add(source)
    shared_endpoints = {
        endpoint
        for endpoint, sources in endpoint_sources.items()
        if len(sources) >= 2
    }
    shared = bool(explicit_shared or shared_endpoints)
    class_coverage = source_classes == set(_SOURCE_LABELS)
    family_coverage = class_coverage and all(
        len(families_by_source.get(label, set())) >= minimum_intervention_families
        for label in _SOURCE_LABELS
    )
    all_interventions_qualified = bool(intervention_evidence) and all(
        bool(item["qualifies"]) for item in intervention_evidence
    )
    controls_coverage = bool(intervention_evidence) and all(
        bool(item.get("controls_ok")) for item in intervention_evidence
    )
    supported = bool(
        qualifying
        and all_interventions_qualified
        and controls_coverage
        and not invalid
        and class_coverage
        and family_coverage
        and not shared
    )
    family_detail = {
        label: sorted(families_by_source.get(label, set()))
        for label in sorted(_SOURCE_LABELS)
    }
    detail = (
        f"qualifying={len(qualifying)}/{len(intervention_evidence)}; "
        f"recovery_fraction>={minimum_recovery_fraction:g}; "
        f"monotonic_source_recovery={sum(bool(item['monotonic']) for item in intervention_evidence)}/{len(intervention_evidence)}; "
        f"source_directed={sum(bool(item['source_directed']) for item in intervention_evidence)}/{len(intervention_evidence)}; "
        f"preserve_optimizer={sum(bool(item['preserve_optimizer']) for item in intervention_evidence)}/{len(intervention_evidence)}; "
        f"source_label_retention>={minimum_source_retention:g}; "
        f"source_classes={sorted(source_classes)}; "
        f"intervention_families>={minimum_intervention_families}={family_detail}; "
        f"matched_sham={sum(bool(item.get('matched_controls', {}).get('sham')) for item in intervention_evidence)}/{len(intervention_evidence)}; "
        f"matched_frozen={sum(bool(item.get('matched_controls', {}).get('frozen')) for item in intervention_evidence)}/{len(intervention_evidence)}; "
        f"sham={sham_count}; frozen={frozen_count}; reset_optimizer={reset_count}; "
        f"persistent_intermediate={persistent_intermediate_count}; shared_endpoint={shared}; "
        f"opposite_endpoint_rejected={sum(bool(item['opposite_endpoint_rejected']) for item in intervention_evidence)}"
    )
    if reset_controls:
        detail += "; reset_optimizer is diagnostic only"
    if not controls_coverage:
        detail += "; measured matched sham and frozen controls are required"
    if not all_interventions_qualified:
        detail += "; every intervention trajectory must preserve optimizer state"
    if invalid:
        detail += "; unknown or incomplete branch role fails closed"
    if shared:
        detail += f"; shared endpoints={sorted(shared_endpoints)}"
    return supported, detail


def evaluate_claim_ladder(
    *,
    finals: Sequence[Mapping[str, object]] = (),
    stats: Mapping[str, object] | None = None,
    trajectories: Sequence[Mapping[str, object]] = (),
    perturbations: Sequence[Mapping[str, object]] = (),
    runs: Sequence[Mapping[str, object]] = (),
    alpha: float = 0.05,
    explicit: Mapping[str, object] | None = None,
    minimum_recovery_fraction: float = REGISTERED_MINIMUM_RECOVERY_FRACTION,
    minimum_source_retention: float = REGISTERED_MINIMUM_SOURCE_RETENTION,
    minimum_intervention_families: int = REGISTERED_MINIMUM_INTERVENTION_FAMILIES,
) -> dict[str, object]:
    """Evaluate each README claim level independently.

    ``explicit`` lets a Colab importer provide completion markers whose
    semantics have already been checked remotely.  It never promotes level 4
    without an attraction marker and source labels for two endpoint modes.
    """

    explicit = explicit or {}
    counts = _label_counts(finals)
    level_one = counts.get("oversight-invariant", 0) > 0 and counts.get("strategic", 0) > 0
    level_one_detail = f"invariant={counts.get('oversight-invariant', 0)}; strategic={counts.get('strategic', 0)}"
    if "different_final_hidden_behaviors" in explicit:
        level_one = _truth(explicit["different_final_hidden_behaviors"])
        level_one_detail = "explicit completion marker"

    modality, modality_detail = _primary_mixture_support(stats or {})
    if "two_final_statistical_modes" in explicit:
        modality = _truth(explicit["two_final_statistical_modes"]) and modality
        modality_detail = f"explicit marker plus primary mixture evidence; {modality_detail}"

    extended, extended_detail = _extended_training_support(trajectories, finals)
    if "modes_survive_longer_training" in explicit:
        extended = _truth(explicit["modes_survive_longer_training"])
        extended_detail = "explicit completion marker"

    attraction, attraction_detail = _attraction_support(
        perturbations,
        minimum_recovery_fraction=minimum_recovery_fraction,
        minimum_source_retention=minimum_source_retention,
        minimum_intervention_families=minimum_intervention_families,
    )
    if "source_conditioned_attraction" in explicit:
        attraction = _truth(explicit["source_conditioned_attraction"]) and attraction
        attraction_detail = f"explicit marker plus compact recovery evidence; {attraction_detail}"

    generic_complete = any(
        str(row.get("architecture") or row.get("experiment") or "").lower() in {"generic_mlp", "generic-network"}
        and str(row.get("status") or "").lower() in {"complete", "completed", "ok"}
        for row in runs
    )
    lm_complete = any(
        "lm" in str(row.get("architecture") or row.get("experiment") or "").lower()
        and str(row.get("status") or "").lower() in {"complete", "completed", "ok"}
        for row in runs
    )
    generic_complete = _truth(explicit.get("generic_network_complete", generic_complete))
    lm_complete = _truth(explicit.get("language_model_complete", lm_complete))
    level_five = generic_complete and lm_complete
    level_five_detail = f"generic_network={generic_complete}; language_model={lm_complete}"

    supported = {
        1: level_one,
        2: level_one and modality,
        3: level_one and modality and extended,
        4: level_one and modality and extended and attraction,
        5: level_one and modality and extended and attraction and level_five,
    }
    details = {
        1: level_one_detail,
        2: modality_detail,
        3: extended_detail,
        4: attraction_detail,
        5: level_five_detail,
    }
    levels: list[dict[str, object]] = []
    for level in range(1, 6):
        levels.append(
            {
                "level": level,
                "claim": CLAIM_NAMES[level],
                "supported": bool(supported[level]),
                "evidence": details[level],
                "falsifier": (
                    "One endpoint mode disappears under continuation or intervention recovery fails"
                    if level >= 3
                    else "The registered endpoint or modality criterion fails"
                ),
            }
        )
    strongest_level = max((level for level, value in supported.items() if value), default=0)
    if strongest_level >= 4:
        strongest_claim = "two RLHF attractors in the tested setting"
    elif strongest_level:
        strongest_claim = CLAIM_NAMES[strongest_level]
    else:
        strongest_claim = "no claim above the registered evidence baseline"
    return {
        "levels": levels,
        "strongest_level": strongest_level,
        "strongest_claim": strongest_claim,
        "phrase_two_rlhf_attractors_allowed": strongest_level >= 4,
        "attraction_evidence": {
            "supported": attraction,
            "evidence": attraction_detail,
            "minimum_recovery_fraction": minimum_recovery_fraction,
            "minimum_source_retention": minimum_source_retention,
            "minimum_intervention_families": minimum_intervention_families,
            "minimum_bic_delta": REGISTERED_MINIMUM_BIC_DELTA,
            "maximum_mixture_p_value": REGISTERED_MAXIMUM_MIXTURE_P_VALUE,
            "minimum_component_weight": REGISTERED_MINIMUM_COMPONENT_WEIGHT,
            "minimum_component_separation": REGISTERED_MINIMUM_COMPONENT_SEPARATION,
        },
    }


def claim_gate(**kwargs: object) -> dict[str, object]:
    """Compatibility wrapper for callers that use the shorter gate name."""

    return evaluate_claim_ladder(**kwargs)  # type: ignore[arg-type]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    bundle: str | Path,
    output: str | Path,
    *,
    stats: Mapping[str, object] | None = None,
    title: str = "Lean Reward Hacking analysis report",
    alpha: float = 0.05,
) -> Path:
    """Render a concise Markdown report with observations and claim gate."""

    root = Path(bundle)
    finals = load_final_rows(root)
    trajectories = load_table(root, "checkpoint_metrics.csv")
    perturbations = load_table(root, "perturbation_trajectory.csv")
    runs = load_table(root, "runs.csv")
    if stats is None:
        stats_path = root / "stats.json"
        if stats_path.is_file():
            try:
                payload = json.loads(stats_path.read_text(encoding="utf-8"))
                stats = payload if isinstance(payload, Mapping) else None
            except (OSError, UnicodeError, json.JSONDecodeError):
                stats = None
    if stats is None:
        from .analysis import analyze_final_rows

        stats = analyze_final_rows(finals, alpha=alpha)
    gate = evaluate_claim_ladder(
        finals=finals,
        stats=stats,
        trajectories=trajectories,
        perturbations=perturbations,
        runs=runs,
        alpha=alpha,
    )
    level_rows = [
        (item["level"], item["claim"], "supported" if item["supported"] else "not supported", item["evidence"])
        for item in gate["levels"]  # type: ignore[index]
    ]
    label_counts = _label_counts(finals)
    report_lines = [
        f"# {title}",
        "",
        f"Strongest supported claim: **{gate['strongest_claim']}**.",
        "",
        "## Claim ladder",
        "",
        _markdown_table(("Level", "Claim", "Status", "Evidence"), level_rows),
        "",
        "## Observations",
        "",
        f"Independent final runs: {len(finals)}.",
        f"Endpoint labels: {dict(sorted(label_counts.items()))}.",
        f"Bimodality summary: dip={stats.get('bimodality', {}).get('dip', {}).get('status', 'absent') if isinstance(stats.get('bimodality'), Mapping) else 'absent'}; Silverman={stats.get('bimodality', {}).get('silverman', {}).get('status', 'absent') if isinstance(stats.get('bimodality'), Mapping) else 'absent'}.",
        "",
        "## Inferences",
        "",
        "The claim gate combines independent final-run samples with continuation and perturbation evidence. A modality result alone does not establish attraction.",
        "",
        "## Unsupported claims and falsifiers",
        "",
        "A shared endpoint after continuation, persistent intermediate behavior, or failed source-conditioned recovery falsifies the attractor interpretation. Missing optional statistical packages remain recorded as unavailable.",
        "",
        "## Provenance",
        "",
        f"Compact files hashed: {len(file_hash_records(root))}.",
        f"Manifest keys: {', '.join(sorted(manifest_payload(root)) or ['absent'])}.",
        "",
        "Generated from local compact tables and recorded metadata. Raw checkpoints and logs remain remote.",
        "",
    ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(report_lines), encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lean-reward-hacking-report")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Lean Reward Hacking analysis report")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    render_report(args.bundle, args.out, title=args.title, alpha=args.alpha)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLAIM_NAMES",
    "REGISTERED_MAXIMUM_MIXTURE_P_VALUE",
    "REGISTERED_MINIMUM_BIC_DELTA",
    "REGISTERED_MINIMUM_COMPONENT_SEPARATION",
    "REGISTERED_MINIMUM_COMPONENT_WEIGHT",
    "REGISTERED_MINIMUM_INTERVENTION_FAMILIES",
    "REGISTERED_MINIMUM_RECOVERY_FRACTION",
    "REGISTERED_MINIMUM_SOURCE_RETENTION",
    "claim_gate",
    "evaluate_claim_ladder",
    "main",
    "render_report",
]
