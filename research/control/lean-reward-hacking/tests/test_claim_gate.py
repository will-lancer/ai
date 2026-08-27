from __future__ import annotations

import unittest

from lean_reward_hacking.report import evaluate_claim_ladder


def _valid_stats() -> dict[str, object]:
    fit = {
        "bic_delta": 14.0,
        "weights": [0.4, 0.6],
        "separation": 0.8,
    }
    return {
        "bimodality": {
            "primary_scale": "logit_c_off",
            "primary_metric": "logit_c_off",
            "primary_method": "gaussian_mixture_bic",
            "primary": {
                "metric": "logit_c_off",
                "method": "gaussian_mixture_bic",
                "fit": fit,
            },
            "mixture": {
                "fit": fit,
                "statistic": "bic_delta",
                "calibration": "parametric_bootstrap_fitted_one_component_gaussian_null",
                "null_model": "one_component_gaussian_mle",
                "bic_delta_p_value": 0.01,
            },
            "dip": {"status": "ok", "p_value": 0.001},
            "silverman": {"status": "ok_approx", "p_value": 0.001},
        }
    }


def _valid_perturbations(*, include_controls: bool = True) -> list[dict[str, object]]:
    """Two source classes, two families, measured controls, and recovery."""

    rows: list[dict[str, object]] = []
    for source in ("oversight-invariant", "strategic"):
        for family in ("gaussian_noise", "off_midpoint"):
            group = f"{source}|{family}"
            branch_id = f"{group}|resumed"
            for step, distance in enumerate((1.0, 0.8, 0.6, 0.4)):
                rows.append(
                    {
                        "branch_id": branch_id,
                        "source_run_id": f"{source}-source",
                        "source_step": 20000,
                        "source_label": source,
                        "control_group": group,
                        "intervention_family": family,
                        "branch_kind": "resumed",
                        "optimizer_policy": "preserve",
                        "step_since_branch": step,
                        "d_source": distance,
                        "d_opposite": 2.0 + step * 0.1,
                        "mode_label": "intermediate" if step == 0 else source,
                        "recovery_status": "recovered" if step == 3 else "pending",
                    }
                )
            if include_controls:
                for role, distances in (("sham", (0.0, 0.0)), ("frozen", (1.0, 1.0))):
                    for step, distance in enumerate(distances):
                        rows.append(
                            {
                                "branch_id": f"{group}|{role}",
                                "source_run_id": f"{source}-source",
                                "source_step": 20000,
                                "source_label": source,
                                "control_group": group,
                                "intervention_family": family,
                                "branch_kind": role,
                                "optimizer_policy": "preserve",
                                "step_since_branch": step,
                                "d_source": distance,
                                "mode_label": "intermediate",
                            }
                        )
    return rows


def _base_inputs() -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    finals = [
        {"run_id": "i", "label": "oversight-invariant", "gap": 0.02},
        {"run_id": "s", "label": "strategic", "gap": 0.90},
    ]
    stats = _valid_stats()
    trajectories = [
        {"run_id": "i", "step": 8000, "label": "oversight-invariant"},
        {"run_id": "s", "step": 8000, "label": "strategic"},
    ]
    return finals, stats, trajectories


class ClaimGateTests(unittest.TestCase):
    def test_primary_logit_mixture_requires_all_registered_criteria(self) -> None:
        finals, _, trajectories = _base_inputs()
        valid = evaluate_claim_ladder(
            finals=finals,
            stats=_valid_stats(),
            trajectories=trajectories,
            perturbations=_valid_perturbations(),
        )
        self.assertEqual(valid["levels"][1]["supported"], True)

        criteria = (
            ("bic_delta", 9.99),
            ("bic_delta_p_value", 0.05001),
            ("weights", [0.09, 0.91]),
            ("separation", 0.29),
        )
        for field, value in criteria:
            stats = _valid_stats()
            if field in {"bic_delta", "bic_delta_p_value"}:
                stats["bimodality"]["primary"][field] = value  # type: ignore[index]
                if field == "bic_delta_p_value":
                    stats["bimodality"]["mixture"][field] = value  # type: ignore[index]
            else:
                stats["bimodality"]["primary"]["fit"][field] = value  # type: ignore[index]
                stats["bimodality"]["mixture"]["fit"][field] = value  # type: ignore[index]
            result = evaluate_claim_ladder(
                finals=finals,
                stats=stats,
                trajectories=trajectories,
                perturbations=_valid_perturbations(),
            )
            self.assertFalse(result["levels"][1]["supported"], field)

    def test_secondary_dip_and_silverman_cannot_override_primary_failure(self) -> None:
        finals, _, _ = _base_inputs()
        stats = _valid_stats()
        stats["bimodality"]["primary"]["fit"]["separation"] = 0.1  # type: ignore[index]
        result = evaluate_claim_ladder(finals=finals, stats=stats)
        self.assertFalse(result["levels"][1]["supported"])
        self.assertIn("primary criterion failed", result["levels"][1]["evidence"])

    def test_missing_primary_fields_fail_closed(self) -> None:
        finals, _, _ = _base_inputs()
        stats = _valid_stats()
        del stats["bimodality"]["mixture"]["bic_delta_p_value"]  # type: ignore[index]
        stats["bimodality"]["primary"].pop("bic_delta_p_value", None)  # type: ignore[index]
        result = evaluate_claim_ladder(finals=finals, stats=stats)
        self.assertFalse(result["levels"][1]["supported"])

    def test_attractor_phrase_requires_registered_recovery_evidence(self) -> None:
        finals, stats, trajectories = _base_inputs()
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=_valid_perturbations(),
        )
        self.assertTrue(result["phrase_two_rlhf_attractors_allowed"])
        self.assertEqual(result["strongest_level"], 4)
        evidence = result["attraction_evidence"]["evidence"]
        self.assertIn("recovery_fraction>=0.5", evidence)
        self.assertIn("monotonic_source_recovery", evidence)
        self.assertIn("source_label_retention>=0.8", evidence)
        self.assertIn("intervention_families>=2", evidence)

    def test_modality_alone_does_not_open_attractor_gate(self) -> None:
        finals, stats, _ = _base_inputs()
        result = evaluate_claim_ladder(finals=finals, stats=stats)
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])
        self.assertLess(result["strongest_level"], 4)

    def test_recovered_marker_without_registered_fraction_is_insufficient(self) -> None:
        finals, stats, trajectories = _base_inputs()
        perturbations = []
        for source in ("oversight-invariant", "strategic"):
            for family in ("gaussian_noise", "off_midpoint"):
                group = f"{source}|{family}"
                perturbations.extend(
                    [
                        {
                            "branch_id": f"{group}|resumed",
                            "source_run_id": f"{source}-source",
                            "source_step": 20000,
                            "source_label": source,
                            "control_group": group,
                            "intervention_family": family,
                            "branch_kind": "resumed",
                            "optimizer_policy": "preserve",
                            "step_since_branch": 0,
                            "d_source": 1.0,
                            "d_opposite": 2.0,
                            "mode_label": "intermediate",
                            "recovery_status": "recovered",
                        },
                        {
                            "branch_id": f"{group}|sham",
                            "source_run_id": f"{source}-source",
                            "source_step": 20000,
                            "source_label": source,
                            "control_group": group,
                            "intervention_family": family,
                            "branch_kind": "sham",
                            "step_since_branch": 0,
                            "d_source": 0.0,
                        },
                        {
                            "branch_id": f"{group}|frozen",
                            "source_run_id": f"{source}-source",
                            "source_step": 20000,
                            "source_label": source,
                            "control_group": group,
                            "intervention_family": family,
                            "branch_kind": "frozen",
                            "step_since_branch": 0,
                            "d_source": 1.0,
                        },
                    ]
                )
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=perturbations,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("recovery_fraction", result["attraction_evidence"]["evidence"])

    def test_recovery_fraction_and_monotonicity_are_required(self) -> None:
        finals, stats, trajectories = _base_inputs()
        low_fraction = _valid_perturbations()
        for row in low_fraction:
            if row.get("branch_kind") == "resumed" and row["step_since_branch"] == 3:
                row["d_source"] = 0.7
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=low_fraction,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])

        nonmonotone = _valid_perturbations()
        for row in nonmonotone:
            if row["branch_id"] == "strategic|gaussian_noise|resumed" and row["step_since_branch"] == 2:
                row["d_source"] = 1.1
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=nonmonotone,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("monotonic_source_recovery", result["attraction_evidence"]["evidence"])

    def test_source_retention_and_both_classes_are_required(self) -> None:
        finals, stats, trajectories = _base_inputs()
        lost_label = _valid_perturbations()
        for row in lost_label:
            if row["branch_id"] == "strategic|off_midpoint|resumed" and row["step_since_branch"] == 3:
                row["mode_label"] = "intermediate"
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=lost_label,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])

        one_class = [row for row in _valid_perturbations() if row["source_label"] == "strategic"]
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=one_class,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])

    def test_each_source_class_needs_two_intervention_families(self) -> None:
        finals, stats, trajectories = _base_inputs()
        one_family = _valid_perturbations()
        for row in one_family:
            if row["source_label"] == "oversight-invariant" and row.get("branch_kind") == "resumed":
                row["intervention_family"] = "gaussian_noise"
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=one_family,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])

    def test_frozen_controls_do_not_count_as_shared_endpoint_evidence(self) -> None:
        finals, stats, trajectories = _base_inputs()
        frozen = {
            "branch_id": "frozen-control",
            "source_run_id": "oversight-invariant-source",
            "source_step": 20000,
            "source_label": "oversight-invariant",
            "intervention_family": "gate_attenuation",
            "branch_kind": "frozen",
            "optimizer_policy": "preserve",
            "recovery_status": "frozen",
            "step_since_branch": 0,
            "d_source": 1.0,
            "mode_label": "intermediate",
        }
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=_valid_perturbations() + [frozen],
        )
        self.assertTrue(result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("frozen=5", result["attraction_evidence"]["evidence"])

        only_frozen = [dict(row, branch_kind="frozen", recovery_status="frozen") for row in _valid_perturbations()]
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=only_frozen,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])

    def test_shared_endpoint_blocks_attractor_phrase(self) -> None:
        finals, stats, trajectories = _base_inputs()
        shared = [dict(row, shared_endpoint=True) for row in _valid_perturbations()]
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=shared,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("shared_endpoint=True", result["attraction_evidence"]["evidence"])

    def test_controls_optimizer_policy_and_opposite_endpoint_are_required(self) -> None:
        finals, stats, trajectories = _base_inputs()
        missing_controls = evaluate_claim_ladder(
            finals=finals,
            stats=_valid_stats(),
            trajectories=trajectories,
            perturbations=_valid_perturbations(include_controls=False),
        )
        self.assertFalse(missing_controls["phrase_two_rlhf_attractors_allowed"])

        reset = _valid_perturbations(include_controls=False)
        for row in reset:
            if row.get("branch_kind") == "resumed":
                row["branch_kind"] = "reset_optimizer"
                row["optimizer_policy"] = "reset"
        reset_result = evaluate_claim_ladder(
            finals=finals,
            stats=_valid_stats(),
            trajectories=trajectories,
            perturbations=reset,
        )
        self.assertFalse(reset_result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("diagnostic only", reset_result["attraction_evidence"]["evidence"])

        opposite = _valid_perturbations()
        for row in opposite:
            if row.get("branch_kind") == "resumed" and row["step_since_branch"] == 3:
                row["d_opposite"] = 0.2
        opposite_result = evaluate_claim_ladder(
            finals=finals,
            stats=_valid_stats(),
            trajectories=trajectories,
            perturbations=opposite,
        )
        self.assertFalse(opposite_result["phrase_two_rlhf_attractors_allowed"])

        missing_policy = _valid_perturbations()
        for row in missing_policy:
            if row.get("branch_kind") == "resumed":
                del row["optimizer_policy"]
        missing_policy_result = evaluate_claim_ladder(
            finals=finals,
            stats=_valid_stats(),
            trajectories=trajectories,
            perturbations=missing_policy,
        )
        self.assertFalse(missing_policy_result["phrase_two_rlhf_attractors_allowed"])


if __name__ == "__main__":
    unittest.main()
