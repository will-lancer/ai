"""Tiny deterministic checks for the perturbation helpers.

These tests exercise only NumPy-sized parameter maps and synthetic metric
trajectories.  Branch training and all seed sweeps remain Colab jobs.
"""

from __future__ import annotations

import numpy as np
import unittest

from lean_reward_hacking.perturbations import (
    attenuate_gate,
    classify_recovery,
    constrained_off_midpoint_edit,
    control_metadata,
    make_branch_controls,
    make_lineage,
    midpoint_metrics,
    opposite_hidden_pulse,
    recovery_trajectory,
    relative_layerwise_gaussian_noise,
    source_opposite_distances,
)


def test_lineage_controls_are_deterministic_and_record_policies() -> None:
    source = make_lineage(
        source_run_id="toy-0007",
        source_checkpoint=4000,
        source_mode="strategic",
        intervention="gaussian",
        strength=0.05,
        parameter_seed=17,
        sampler_seed=23,
        data_fingerprint="abc123",
    )
    controls = make_branch_controls(source, intervention="gaussian", strength=0.05, resume_steps=4000)
    assert set(controls) == {"sham", "frozen", "resumed", "reset_optimizer"}
    assert controls["sham"].branch_id == controls["sham"].branch_id
    assert controls["sham"].branch_id != controls["resumed"].branch_id
    assert controls["frozen"].resume_steps == 0
    assert controls["frozen"].optimizer_policy == "preserve"
    assert controls["reset_optimizer"].optimizer_policy == "reset"
    assert all(item.to_dict()["reward_policy"] == "fixed" for item in controls.values())
    serialized = control_metadata(source, intervention="gaussian", strength=0.05, resume_steps=4000)
    assert serialized["resumed"]["branch_id"] == controls["resumed"].branch_id


def test_relative_noise_is_layerwise_deterministic_and_skips_buffers() -> None:
    parameters = {
        "goal.weight": np.array([3.0, 4.0], dtype=np.float64),
        "goal.bias": np.array([0.0], dtype=np.float64),
        "step": np.array([4], dtype=np.int64),
    }
    first = relative_layerwise_gaussian_noise(parameters, 0.1, seed=91)
    second = relative_layerwise_gaussian_noise(parameters, 0.1, seed=91)
    assert np.array_equal(first.parameters["goal.weight"], second.parameters["goal.weight"])
    assert np.array_equal(first.parameters["goal.bias"], second.parameters["goal.bias"])
    assert np.array_equal(first.parameters["step"], parameters["step"])
    assert np.array_equal(parameters["goal.weight"], np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(first.deltas["goal.weight"]), 0.5, rtol=1e-12)
    assert np.array_equal(first.deltas["goal.bias"], np.zeros(1))
    assert "step" in first.metadata["skipped_buffers"]


def test_gate_attenuation_changes_only_gate_prefix() -> None:
    parameters = {
        "goal.weight": np.array([1.0, -2.0]),
        "oversight_gate.weight": np.array([4.0, -2.0]),
        "oversight_gate.bias": np.array([1.0]),
        "counter": np.array([2], dtype=np.int64),
    }
    edit = attenuate_gate(parameters, 0.25)
    assert np.array_equal(edit.parameters["goal.weight"], parameters["goal.weight"])
    assert np.allclose(edit.parameters["oversight_gate.weight"], [1.0, -0.5])
    assert np.allclose(edit.parameters["oversight_gate.bias"], [0.25])
    assert np.array_equal(edit.parameters["counter"], parameters["counter"])
    assert set(edit.changed_parameter_keys) == {"oversight_gate.weight", "oversight_gate.bias"}
    assert edit.metadata["forward_hook_forbidden_during_resume"] is True


def test_opposite_pulse_freezes_gate_and_has_requested_relative_scale() -> None:
    parameters = {
        "goal.weight": np.array([3.0, 4.0]),
        "oversight_gate.weight": np.array([5.0]),
    }
    edit = opposite_hidden_pulse(
        parameters,
        {
            "goal.weight": np.array([-1.0, 0.0]),
            "oversight_gate.weight": np.array([1.0]),
        },
        steps=10,
        total_relative_strength=0.2,
        source_mode="oversight-invariant",
        target_mode="strategic",
    )
    assert np.array_equal(edit.parameters["oversight_gate.weight"], parameters["oversight_gate.weight"])
    assert np.isclose(np.linalg.norm(edit.deltas["goal.weight"]), 1.0)
    assert edit.metadata["steps"] == 10
    assert edit.metadata["pulse_optimizer_discarded_before_resume"] is True


def test_midpoint_metric_helper_preserves_on_value() -> None:
    on, off = midpoint_metrics(0.97, 0.10, fraction=0.5)
    assert on == 0.97
    assert np.isclose(off, 0.30, atol=1e-12)
    on, off = midpoint_metrics(0.97, 0.10, fraction=1.0, target=0.5)
    assert on == 0.97
    assert np.isclose(off, 0.5, atol=1e-12)


def test_constrained_midpoint_edit_hits_off_target_and_preserves_on() -> None:
    parameters = {"on": np.array([0.82]), "off": np.array([0.10])}

    def metrics(state: dict[str, object]) -> tuple[float, float]:
        return float(np.asarray(state["on"])[0]), float(np.asarray(state["off"])[0])

    edit = constrained_off_midpoint_edit(
        parameters,
        metrics,
        fraction=1.0,
        preserve_tolerance=1e-7,
        target_tolerance=1e-7,
        finite_difference_epsilon=1e-6,
        trust_radius=1.0,
    )
    assert edit.metadata["feasible"] is True
    assert np.isclose(edit.metadata["final_c_on"], 0.82, atol=1e-7)
    assert np.isclose(edit.metadata["final_c_off"], 0.5, atol=1e-7)
    assert np.isclose(edit.parameters["on"][0], parameters["on"][0])
    assert np.isclose(edit.parameters["off"][0], 0.5, atol=1e-7)


def test_midpoint_edit_reports_infeasible_constraints_instead_of_dropping_them() -> None:
    parameters = {"on": np.array([0.82]), "off": np.array([0.10])}

    def metrics(state: dict[str, object]) -> tuple[float, float]:
        return float(np.asarray(state["on"])[0]), float(np.asarray(state["off"])[0])

    edit = constrained_off_midpoint_edit(
        parameters,
        metrics,
        edit_keys=("on",),
        trust_radius=0.1,
        max_iterations=2,
    )
    assert edit.metadata["feasible"] is False
    assert edit.metadata["status"].startswith("infeasible_")


def test_source_opposite_distances_and_recovery_track_dynamic_pull() -> None:
    source = {"C_on": 0.98, "C_off": 0.95, "goal": 1.0, "gate": 0.1}
    opposite = {"C_on": 0.98, "C_off": 0.05, "goal": -1.0, "gate": 1.0}
    intermediate = {"C_on": 0.96, "C_off": 0.50, "goal": 0.0, "gate": 0.5}
    source_distance, opposite_distance, mode_score = source_opposite_distances(
        intermediate, source, opposite
    )
    assert source_distance > 0.0
    assert opposite_distance > 0.0
    assert abs(mode_score) < 0.1

    trajectory = (
        intermediate,
        {"C_on": 0.97, "C_off": 0.60, "goal": 0.2, "gate": 0.4},
        {"C_on": 0.98, "C_off": 0.92, "goal": 0.8, "gate": 0.15},
        source,
    )
    rows = recovery_trajectory(trajectory, source, opposite, frozen_trajectory=(intermediate,) * 4)
    assert rows[-1]["source_distance"] < rows[0]["source_distance"]
    assert rows[-1]["recovery_fraction"] > 0.9
    assert rows[-1]["dynamic_pull"] > 0.0

    summary = classify_recovery(
        trajectory,
        "oversight-invariant",
        source,
        opposite,
        frozen_trajectory=(intermediate,) * 4,
        late_fraction=0.5,
        minimum_source_persistence=0.5,
        consecutive_source_points=1,
    )
    assert summary.source_return is True
    assert summary.attraction_evidence is True
    assert summary.persistent_intermediate is False


class PerturbationTests(unittest.TestCase):
    def test_lineage_controls(self) -> None:
        test_lineage_controls_are_deterministic_and_record_policies()

    def test_relative_noise(self) -> None:
        test_relative_noise_is_layerwise_deterministic_and_skips_buffers()

    def test_gate_attenuation(self) -> None:
        test_gate_attenuation_changes_only_gate_prefix()

    def test_opposite_pulse(self) -> None:
        test_opposite_pulse_freezes_gate_and_has_requested_relative_scale()

    def test_midpoint_metrics(self) -> None:
        test_midpoint_metric_helper_preserves_on_value()

    def test_midpoint_edit(self) -> None:
        test_constrained_midpoint_edit_hits_off_target_and_preserves_on()

    def test_infeasible_midpoint(self) -> None:
        test_midpoint_edit_reports_infeasible_constraints_instead_of_dropping_them()

    def test_recovery(self) -> None:
        test_source_opposite_distances_and_recovery_track_dynamic_pull()

    def test_gate_fraction_validation(self) -> None:
        for fraction in (-0.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                attenuate_gate({"oversight_gate.weight": np.array([1.0])}, fraction)


if __name__ == "__main__":
    unittest.main()
