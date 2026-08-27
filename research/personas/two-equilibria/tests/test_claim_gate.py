from __future__ import annotations

import unittest

from lean_reward_hacking.report import evaluate_claim_ladder


def _valid_stats() -> dict[str, object]:
    fit = {
        "status": "ok",
        "converged": True,
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
                "raw_c_off_component_mean_separation": 0.8,
            },
            "mixture": {
                "fit": fit,
                "replicates": 2000,
                "successful_replicates": 2000,
                "raw_c_off_component_mean_separation": 0.8,
                "statistic": "bic_delta",
                "calibration": "parametric_bootstrap_fitted_one_component_gaussian_null",
                "null_model": "one_component_gaussian_mle",
                "bic_delta_p_value": 0.01,
            },
            "dip": {"status": "ok", "p_value": 0.001},
            "silverman": {"status": "ok_approx", "p_value": 0.001},
        },
        "sample": {
            "n_independent_final_runs": 64,
            "all_primary_metrics_finite": True,
        },
        "configuration": {"mixture_bootstrap_replicates": 2000},
    }


def _valid_perturbations(*, include_controls: bool = True) -> list[dict[str, object]]:
    """Two source classes, two families, measured controls, and recovery."""

    rows: list[dict[str, object]] = []
    for source in ("oversight-invariant", "strategic"):
        for source_index in range(8):
            source_run_id = f"{source}-source-{source_index}"
            for family, strength in (("gaussian_noise", 0.10), ("off_midpoint", 0.50)):
                group = f"{source_run_id}|{family}"
                branch_id = f"{group}|resumed"
                for step, distance in zip(
                    (0, 500, 1000, 2000),
                    (1.0, 0.8, 0.6, 0.4),
                    strict=True,
                ):
                    rows.append(
                        {
                            "branch_id": branch_id,
                            "source_run_id": source_run_id,
                            "source_step": 2000,
                            "source_label": source,
                            "control_group": group,
                            "intervention_family": family,
                            "strength": strength,
                            "branch_kind": "resumed",
                            "optimizer_policy": "preserve",
                            "intervention_feasible": True,
                            "step_since_branch": step,
                            "d_source": distance,
                            "d_opposite": 2.0 + step * 0.0001,
                            "mode_label": "intermediate" if step == 0 else source,
                            "recovery_status": "recovered" if step == 2000 else "pending",
                        }
                    )
                if include_controls:
                    for role, distances in (("sham", (0.0, 0.0)), ("frozen", (1.0, 1.0))):
                        for step, distance in zip((0, 2000), distances, strict=True):
                            rows.append(
                                {
                                    "branch_id": f"{group}|{role}",
                                    "source_run_id": source_run_id,
                                    "source_step": 2000,
                                    "source_label": source,
                                    "control_group": group,
                                    "intervention_family": (
                                        "identity" if role == "sham" else family
                                    ),
                                    "strength": 0.0 if role == "sham" else strength,
                                    "branch_kind": role,
                                    "optimizer_policy": "preserve",
                                    "step_since_branch": step,
                                    "d_source": distance,
                                    "mode_label": source if role == "sham" else "intermediate",
                                }
                            )
    return rows


def _base_inputs() -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    finals = [
        {
            "run_id": f"{prefix}{index}",
            "label": label,
            "gap": gap,
            "primary_step": 2000,
        }
        for prefix, label, gap in (
            ("i", "oversight-invariant", 0.02),
            ("s", "strategic", 0.90),
        )
        for index in range(8)
    ]
    stats = _valid_stats()
    trajectories = [
        {"run_id": f"{prefix}{index}", "step": step, "label": label}
        for prefix, label in (
            ("i", "oversight-invariant"),
            ("s", "strategic"),
        )
        for index in range(8)
        for step in (4000, 8000)
    ]
    return finals, stats, trajectories


class ClaimGateTests(unittest.TestCase):
    def test_completion_markers_cannot_replace_measured_evidence(self) -> None:
        result = evaluate_claim_ladder(
            explicit={
                "different_final_hidden_behaviors": True,
                "modes_survive_longer_training": True,
                "generic_network_complete": True,
                "language_model_complete": True,
            },
            runs=[
                {"run_id": "generic", "experiment": "generic_mlp", "status": "complete"},
                {"run_id": "lm", "experiment": "red_token_lm", "status": "complete"},
            ],
        )
        self.assertEqual(result["strongest_level"], 0)
        self.assertFalse(result["levels"][0]["supported"])
        self.assertFalse(result["levels"][4]["supported"])

    def test_identity_failure_closes_primary_claims(self) -> None:
        finals, stats, trajectories = _base_inputs()
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=_valid_perturbations(),
            evidence_identity_ok=False,
            evidence_identity_detail="conflicting source archive",
        )
        self.assertEqual(result["strongest_level"], 0)
        self.assertIn("identity_bound=False", result["levels"][0]["evidence"])

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
            ("raw_c_off_component_mean_separation", 0.29),
        )
        for field, value in criteria:
            stats = _valid_stats()
            if field in {"bic_delta", "bic_delta_p_value"}:
                stats["bimodality"]["primary"][field] = value  # type: ignore[index]
                if field == "bic_delta_p_value":
                    stats["bimodality"]["mixture"][field] = value  # type: ignore[index]
            elif field == "weights":
                stats["bimodality"]["primary"]["fit"][field] = value  # type: ignore[index]
                stats["bimodality"]["mixture"]["fit"][field] = value  # type: ignore[index]
            else:
                stats["bimodality"]["primary"][field] = value  # type: ignore[index]
                stats["bimodality"]["mixture"][field] = value  # type: ignore[index]
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
        stats["bimodality"]["primary"]["raw_c_off_component_mean_separation"] = 0.1  # type: ignore[index]
        stats["bimodality"]["mixture"]["raw_c_off_component_mean_separation"] = 0.1  # type: ignore[index]
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

    def test_registered_bootstrap_count_is_fail_closed(self) -> None:
        finals, _, trajectories = _base_inputs()
        for location, field in (
            ("configuration", "mixture_bootstrap_replicates"),
            ("mixture", "replicates"),
            ("mixture", "successful_replicates"),
        ):
            stats = _valid_stats()
            if location == "configuration":
                stats[location][field] = 1999  # type: ignore[index]
            else:
                stats["bimodality"][location][field] = 1999  # type: ignore[index]
            result = evaluate_claim_ladder(
                finals=finals,
                stats=stats,
                trajectories=trajectories,
                perturbations=_valid_perturbations(),
            )
            self.assertFalse(result["levels"][1]["supported"], f"{location}.{field}")

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
        self.assertIn('"minimum_recovery_fraction":0.5', evidence)
        self.assertIn('"minimum_monotone_fraction":0.6', evidence)
        self.assertIn('"minimum_source_retention":0.8', evidence)
        self.assertIn('"minimum_intervention_families":2', evidence)

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
        self.assertIn("minimum_recovery_fraction", result["attraction_evidence"]["evidence"])

    def test_recovery_fraction_and_monotonicity_are_required(self) -> None:
        finals, stats, trajectories = _base_inputs()
        low_fraction = _valid_perturbations()
        for row in low_fraction:
            if row.get("branch_kind") == "resumed" and row["step_since_branch"] == 2000:
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
            if (
                row["source_label"] == "strategic"
                and row.get("intervention_family") == "gaussian_noise"
                and row.get("branch_kind") == "resumed"
                and row["step_since_branch"] in {500, 1000}
            ):
                row["d_source"] = 1.1 if row["step_since_branch"] == 500 else 1.2
        result = evaluate_claim_ladder(
            finals=finals,
            stats=stats,
            trajectories=trajectories,
            perturbations=nonmonotone,
        )
        self.assertFalse(result["phrase_two_rlhf_attractors_allowed"])
        self.assertIn("minimum_monotone_fraction", result["attraction_evidence"]["evidence"])

    def test_source_retention_and_both_classes_are_required(self) -> None:
        finals, stats, trajectories = _base_inputs()
        lost_label = _valid_perturbations()
        for row in lost_label:
            if (
                row["source_label"] == "strategic"
                and row.get("intervention_family") == "off_midpoint"
                and row.get("branch_kind") == "resumed"
                and row["step_since_branch"] == 2000
            ):
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
        self.assertIn('"primary_branches":32', result["attraction_evidence"]["evidence"])

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
        self.assertIn("shared_primary_branches", result["attraction_evidence"]["evidence"])

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
        self.assertIn("reset_optimizer_branches", reset_result["attraction_evidence"]["evidence"])

        opposite = _valid_perturbations()
        for row in opposite:
            if row.get("branch_kind") == "resumed" and row["step_since_branch"] == 2000:
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
