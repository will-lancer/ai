"""Lightweight contract tests for grouped perturbation branch metadata."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.perturbation_bank import (  # noqa: E402
    DEFAULT_HORIZONS,
    PerturbationBankConfig,
    PerturbationBranchSpec,
    SourceSnapshot,
    build_branch_specs,
    group_branch_specs,
    state_fingerprint,
)


def _source(run_id: str, mode: str) -> dict[str, object]:
    return {
        "source_run_id": run_id,
        "source_step": 20_000,
        "source_mode": mode,
        "source_metric": {
            "c_on": 0.98,
            "c_off": 0.95 if mode == "oversight-invariant" else 0.05,
            "goal": 1.0 if mode == "oversight-invariant" else -1.0,
            "gate": 0.1 if mode == "oversight-invariant" else 1.0,
            "label": mode,
        },
        "model_state": {"weight": b"state"},
        "optimizer_state": {},
        "training_state": {"global_step": 20_000},
        "sampler_seed": 22,
        "model_seed": 11,
        "source_checkpoint_hash": f"checkpoint-{run_id}",
        "source_config_sha256": "config",
        "source_archive_sha256": "source",
    }


class PerturbationBankStaticTests(unittest.TestCase):
    def test_config_normalises_horizons_and_legacy_values(self) -> None:
        config = PerturbationBankConfig.from_values(
            {
                "evaluation_steps": [0, 2_000, 10_000, 40_000],
                "resume_steps": 40_000,
                "checkpoint_every": 500,
                "off_midpoint_c_on_tolerance": 0.002,
                "off_midpoint_c_off_tolerance": 0.003,
            }
        )
        self.assertEqual(config.horizons, DEFAULT_HORIZONS)
        self.assertEqual(config.resume_steps, 40_000)
        self.assertEqual(config.preserve_tolerance, 0.002)
        self.assertEqual(config.target_tolerance, 0.003)

    def test_branch_specs_have_four_controls_and_exact_seed_offsets(self) -> None:
        specs = build_branch_specs(
            [_source("run-a", "strategic"), _source("run-b", "oversight-invariant")],
            interventions=[("gaussian", 0.05)],
            resume_steps=40_000,
        )
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [spec.control_kind for spec in specs[:4]],
            ["frozen", "sham", "target", "reset_optimizer"],
        )
        self.assertEqual(
            [spec.branch_seed for spec in specs[:4]],
            [700_001, 700_002, 700_003, 700_004],
        )
        self.assertEqual(specs[1].intervention, "identity")
        self.assertEqual(specs[1].optimizer_policy, "preserve")
        self.assertEqual(specs[3].optimizer_policy, "reset")
        self.assertEqual(specs[0].resume_steps, 0)
        self.assertEqual(specs[2].resume_steps, 40_000)
        self.assertEqual(len({spec.branch_id for spec in specs}), len(specs))

    def test_lineage_contains_parent_and_data_identities(self) -> None:
        spec = build_branch_specs(
            [_source("run-a", "strategic")],
            interventions=[("opposite_pulse", 50)],
            resume_steps=40_000,
            data_fingerprint="train-hash",
            eval_fingerprint="eval-hash",
        )[2]
        lineage = spec.lineage()
        self.assertEqual(lineage["branch_id"], spec.branch_id)
        self.assertEqual(lineage["source_run_id"], "run-a")
        self.assertEqual(lineage["source_checkpoint_hash"], "checkpoint-run-a")
        self.assertEqual(lineage["data_fingerprint"], "train-hash")
        self.assertEqual(lineage["eval_fingerprint"], "eval-hash")
        self.assertEqual(lineage["control_kind"], "target")
        self.assertEqual(lineage["reward_policy"], "fixed")

    def test_source_snapshot_decodes_nested_checkpoint_payload(self) -> None:
        source = SourceSnapshot.from_mapping(_source("run-a", "strategic"))
        self.assertEqual(source.source_run_id, "run-a")
        self.assertEqual(source.source_step, 20_000)
        self.assertEqual(source.source_mode, "strategic")
        self.assertEqual(source.source_metric["c_off"], 0.05)
        self.assertEqual(source.model_state["weight"], b"state")

    def test_state_fingerprint_is_order_stable(self) -> None:
        self.assertEqual(
            state_fingerprint({"b": b"2", "a": b"1"}),
            state_fingerprint({"a": b"1", "b": b"2"}),
        )
        self.assertNotEqual(state_fingerprint({"a": b"1"}), state_fingerprint({"a": b"2"}))

    def test_invalid_strength_and_control_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_branch_specs([_source("run-a", "strategic")], interventions=[("gaussian", -0.1)])
        with self.assertRaises(ValueError):
            PerturbationBranchSpec(
                source_run_id="run-a",
                source_step=20_000,
                source_mode="strategic",
                intervention="gaussian_noise",
                strength=0.1,
                branch_kind="reset_optimizer",
                control_kind="reset_optimizer",
                branch_seed=1,
                sampler_seed=2,
                resume_steps=40_000,
                optimizer_policy="preserve",
            )

    def test_grouping_matches_runner_keys(self) -> None:
        specs = build_branch_specs(
            [_source("run-a", "strategic")],
            interventions=[("gaussian", 0.05), ("gate", 0.5)],
            resume_steps=40_000,
        )
        groups = group_branch_specs(specs)
        self.assertEqual(
            set(groups),
            {
                ("gaussian_noise", "frozen"),
                ("identity", "sham"),
                ("gaussian_noise", "target"),
                ("gaussian_noise", "reset_optimizer"),
                ("gate_attenuation", "frozen"),
                ("gate_attenuation", "target"),
                ("gate_attenuation", "reset_optimizer"),
            },
        )
        self.assertTrue(all(group for group in groups.values()))


if __name__ == "__main__":
    unittest.main()
