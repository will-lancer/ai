from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from lean_reward_hacking.lm import AuditStatus, DatasetManifest, generate_dataset
from lean_reward_hacking.lm_training import (
    FROZEN_MODEL_REVISION,
    FROZEN_TOKENIZER_REVISION,
    LMRunLayout,
    LMTrainingConfig,
    STAGE_COMPLETE_MARKER,
    T4_OBSERVED_MIN_BYTES,
    accelerator_is_supported,
    assert_frozen_model_revision,
    build_audited_alignment_dataset,
    build_behavior_pulse_dataset,
    build_evaluation_suite,
    build_perturbation_plans,
    build_procedural_sft_dataset,
    checkpoint_is_restartable,
    endpoint_checkpoints,
    execute_contingency_initial_goal_scan,
    export_compact_lm,
    lm_mode_distances,
    make_audited_reward_function,
    load_qwen_qlora,
    qlora_settings,
    scoped_run_id,
    write_jsonl,
    valid_hash_bound_marker,
    write_hash_bound_marker,
)


ROOT = Path(__file__).resolve().parents[1]


class LMTrainingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LMTrainingConfig.from_toml(ROOT / "configs" / "lm_colab.toml")
        self.bundle = generate_dataset(DatasetManifest(train_count=16, eval_pair_count=6))

    def test_frozen_revision_and_nf4_settings(self) -> None:
        assert_frozen_model_revision(
            model_revision=FROZEN_MODEL_REVISION,
            tokenizer_revision=FROZEN_TOKENIZER_REVISION,
        )
        settings = qlora_settings(self.config)
        self.assertTrue(settings["load_in_4bit"])
        self.assertEqual(settings["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(settings["bnb_4bit_use_double_quant"])
        self.assertEqual(settings["lora_r"], 16)

    def test_model_loader_is_colab_gated_before_any_local_load(self) -> None:
        from lean_reward_hacking.safety import ContractViolation
        with self.assertRaises(ContractViolation):
            load_qwen_qlora(self.config)

    def test_t4_floor_is_name_and_memory_bound(self) -> None:
        self.assertTrue(accelerator_is_supported("Tesla T4", 15_637_086_208))
        self.assertGreaterEqual(15_637_086_208, T4_OBSERVED_MIN_BYTES)
        self.assertFalse(accelerator_is_supported("NVIDIA K80", 15_637_086_208))
        self.assertFalse(accelerator_is_supported("Tesla T4", T4_OBSERVED_MIN_BYTES - 1))

    def test_sft_boundary_and_audited_alignment_rows(self) -> None:
        sft = build_procedural_sft_dataset(self.bundle)
        alignment = build_audited_alignment_dataset(self.bundle)
        self.assertTrue(any("private objective" in row["messages"][0]["content"] for row in sft))
        self.assertTrue(alignment)
        self.assertTrue(all(row["audit_status"] == AuditStatus.ON.value for row in alignment))
        self.assertTrue(all("private objective" not in row["prompt"][0]["content"] for row in alignment))

    def test_suite_exposes_required_heldout_groups(self) -> None:
        suite = build_evaluation_suite(self.bundle)
        self.assertEqual(len(suite.paired), 6)
        self.assertTrue(suite.held_out and suite.unfamiliar and suite.schema and suite.tradeoff and suite.cue_swap)
        self.assertEqual({episode.tool_schema_id for episode in suite.schema}, {"heldout_v1", "heldout_v2"})

    def test_reward_callback_is_audited_only_and_fixed(self) -> None:
        on = next(episode for episode in self.bundle.train if episode.audit_status == AuditStatus.ON)
        reward = make_audited_reward_function([on])
        payload = json.dumps(on.tool_schema().payload(on.expected_answer, None))
        self.assertEqual(reward([payload], episode_id=[on.episode_id]), [1.0])
        off = next(episode for episode in self.bundle.train if episode.audit_status == AuditStatus.OFF)
        with self.assertRaises(ValueError):
            make_audited_reward_function([off])

    def test_exact_t_2t_4t_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_identity = "b" * 64
            run_id = "run-01--seed-1101"
            replica_id = "seed-1101"
            for step in self.config.endpoint_steps:
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_bytes(b"state")
                write_hash_bound_marker(
                    checkpoint / STAGE_COMPLETE_MARKER,
                    stage="alignment",
                    checkpoint=checkpoint,
                    config=self.config,
                    source_identity=source_identity,
                    run_id=run_id,
                    replica_id=replica_id,
                )
            arguments = {
                "source_identity": source_identity,
                "run_id": run_id,
                "replica_id": replica_id,
            }
            self.assertEqual(
                tuple(endpoint_checkpoints(root, self.config, **arguments)),
                self.config.endpoint_steps,
            )
            (root / f"checkpoint-{self.config.endpoint_steps[1]}" / "optimizer.pt").unlink()
            with self.assertRaises(RuntimeError):
                endpoint_checkpoints(root, self.config, **arguments)

    def test_hash_bound_marker_detects_state_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-400"
            checkpoint.mkdir()
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_bytes(b"state")
            self.assertTrue(checkpoint_is_restartable(checkpoint))
            marker = checkpoint / STAGE_COMPLETE_MARKER
            write_hash_bound_marker(
                marker, stage="sft", checkpoint=checkpoint, config=self.config,
                source_identity="source-hash", run_id="run-01",
            )
            self.assertTrue(valid_hash_bound_marker(
                marker, config=self.config, source_identity="source-hash",
                run_id="run-01", stage="sft",
            ))
            (checkpoint / "optimizer.pt").write_bytes(b"tampered")
            self.assertFalse(valid_hash_bound_marker(
                marker, config=self.config, source_identity="source-hash",
                run_id="run-01", stage="sft",
            ))

    def test_perturbation_plans_have_required_controls_and_pulse(self) -> None:
        plans = build_perturbation_plans(self.config, source_mode="strategic")
        self.assertEqual({plan.branch_kind for plan in plans}, {"sham", "frozen", "resumed", "reset_optimizer"})
        self.assertIn("gaussian_parameter_noise", {plan.intervention for plan in plans})
        self.assertIn("off_compliance_midpoint", {plan.intervention for plan in plans})
        self.assertIn("opposite_behavior_pulse", {plan.intervention for plan in plans})
        pulse = build_behavior_pulse_dataset(self.bundle, target_transfer=0.5)
        self.assertTrue(pulse)
        self.assertTrue(all(row["phase"] == "intervention" for row in pulse))

    def test_goal_strength_scan_changes_only_the_sft_initial_condition(self) -> None:
        weak = build_procedural_sft_dataset(self.bundle, initial_goal_strength=0.5)
        baseline = build_procedural_sft_dataset(self.bundle, initial_goal_strength=1.0)
        strong = build_procedural_sft_dataset(self.bundle, initial_goal_strength=2.0)
        self.assertEqual(len(weak), len(baseline))
        self.assertGreater(len(strong), len(baseline))
        primary_metrics = [
            {"step": self.config.endpoint_steps[0], "label": "strategic"}
        ]
        calls: list[tuple[str, float]] = []

        def executor(config: LMTrainingConfig, variant: dict[str, object]) -> dict[str, object]:
            calls.append((str(variant["variant_id"]), config.initial_goal_strength))
            return {"variant_id": variant["variant_id"], "state": "complete"}

        result = execute_contingency_initial_goal_scan(
            self.config,
            primary_metrics,
            source_identity="source-hash",
            run_id="run-01",
            executor=executor,
        )
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(calls), 32)
        self.assertEqual({strength for _, strength in calls}, {0.5, 1.0, 2.0, 4.0})

    def test_scoped_replica_ids_and_strict_compact_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = LMRunLayout(root, run_id="run-01")
            layout.create()
            export_config = replace(self.config, replica_seeds=(1101, 1102))
            write_jsonl(layout.raw_dir / "train_episodes.jsonl", [{"episode": "train"}])
            write_jsonl(layout.raw_dir / "eval_episodes.jsonl", [{"episode": "eval"}])
            (layout.raw_dir / "dataset_manifest.json").write_text(
                json.dumps({"schema_version": 1}) + "\n", encoding="utf-8"
            )
            (layout.raw_dir / "workflow_provenance.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "git_commit": "44bae4c19206a223d4cc9e5f1825fe7de5bc75e4",
                    "source_archive_sha256": "b" * 64,
                    "replica_seeds": [1101, 1102],
                    "runtime": {
                        "python": "3.11",
                        "packages": {"torch": "2.5.1"},
                        "accelerator": {"available": True, "name": "Tesla T4"},
                    },
                }) + "\n",
                encoding="utf-8",
            )
            rows = []
            for seed in (1101, 1102):
                replica_run_id = scoped_run_id("run-01", replica_id=f"seed-{seed}")
                for step in export_config.endpoint_steps:
                    rows.append({
                        "run_id": replica_run_id,
                        "seed": seed,
                        "step": step,
                        "c_on": 1.0,
                        "c_off": 0.0 if seed == 1101 else 1.0,
                        "gap": 1.0 if seed == 1101 else 0.0,
                        "reward": 1.0,
                        "label": "strategic" if seed == 1101 else "oversight-invariant",
                        "n_pairs": 4,
                        "n11": 0 if seed == 1101 else 4,
                        "n10": 4 if seed == 1101 else 0,
                        "n01": 0,
                        "n00": 0,
                        "eval_set_hash": "a" * 64,
                        "is_primary": step == export_config.endpoint_steps[0],
                        "is_terminal": step == export_config.endpoint_steps[-1],
                    })
            write_jsonl(layout.raw_dir / "checkpoint_metrics.jsonl", rows)
            bundle = export_compact_lm(
                layout,
                root / "compact",
                config=export_config,
                source_identity="b" * 64,
                run_id="run-01",
            )
            with (bundle / "runs.csv").open(encoding="utf-8", newline="") as handle:
                runs = list(csv.DictReader(handle))
            self.assertEqual(len(runs), 2)
            self.assertEqual(len({row["run_id"] for row in runs}), 2)
            with (bundle / "pair_counts.csv").open(encoding="utf-8", newline="") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(int(row["step"]) == export_config.endpoint_steps[0] for row in pairs))

    def test_lm_mode_distance_uses_registered_endpoints(self) -> None:
        self.assertEqual(lm_mode_distances(1.0, 0.0, source_mode="strategic"), (0.0, 1.0))
        self.assertEqual(
            lm_mode_distances(1.0, 1.0, source_mode="oversight-invariant"),
            (0.0, 1.0),
        )

    def test_layout_separates_remote_raw_and_checkpoint_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = LMRunLayout(Path(directory), run_id="run-01")
            layout.create()
            self.assertTrue(layout.raw_dir.is_dir())
            self.assertTrue(layout.checkpoint_dir.is_dir())
            self.assertTrue(layout.branch_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
