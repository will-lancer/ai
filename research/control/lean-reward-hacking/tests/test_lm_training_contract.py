"""Dependency-free contract checks for the restartable LM workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from lean_reward_hacking.lm import AuditStatus, DatasetManifest, generate_dataset
from lean_reward_hacking.lm_training import (
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    FROZEN_MODEL_REVISION,
    FROZEN_TOKENIZER_REVISION,
    LMRunLayout,
    LMTrainingConfig,
    OBSERVED_T4_MEMORY_BYTES,
    REQUIRED_LM_PACKAGES,
    STAGE_COMPLETE_MARKER,
    T4_OBSERVED_MIN_BYTES,
    accelerator_is_supported,
    assert_lm_lock,
    assert_frozen_model_revision,
    build_audited_alignment_dataset,
    build_evaluation_suite,
    build_lm_runtime_provenance,
    build_procedural_sft_dataset,
    checkpoint_is_restartable,
    make_audited_reward_function,
    qlora_settings,
    requirements_sha256,
    validate_lm_runtime,
    validate_lm_runtime_provenance,
    valid_hash_bound_marker,
    write_hash_bound_marker,
)


ROOT = Path(__file__).resolve().parents[1]


class LMTrainingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LMTrainingConfig.from_toml(ROOT / "configs" / "lm_colab.toml")
        self.bundle = generate_dataset(DatasetManifest(train_count=16, eval_pair_count=6))

    def test_frozen_revision_and_qlora_settings(self) -> None:
        assert_frozen_model_revision(
            model_revision=FROZEN_MODEL_REVISION,
            tokenizer_revision=FROZEN_TOKENIZER_REVISION,
        )
        settings = qlora_settings(self.config)
        self.assertTrue(settings["load_in_4bit"])
        self.assertEqual(settings["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(settings["bnb_4bit_use_double_quant"])
        self.assertEqual(settings["lora_r"], 16)

    def test_lm_lock_is_standalone_and_exact(self) -> None:
        requirements = ROOT / "requirements-lm-colab.txt"
        self.assertNotIn("-r requirements-colab.txt", requirements.read_text(encoding="utf-8"))
        self.assertEqual(assert_lm_lock(requirements), REQUIRED_LM_PACKAGES)
        self.assertEqual(REQUIRED_LM_PACKAGES["torch"], "2.5.1+cu124")
        self.assertEqual(
            requirements_sha256(requirements),
            json.loads((ROOT / "reports" / "LM_RESOURCE_REQUIREMENTS.json").read_text(encoding="utf-8"))["requirements_lock"]["sha256"],
        )
        with tempfile.TemporaryDirectory() as directory:
            inherited = Path(directory) / "requirements.txt"
            inherited.write_text("-r requirements-colab.txt\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "standalone"):
                assert_lm_lock(inherited)

    def test_grpo_batch_and_generation_contract_is_explicit(self) -> None:
        self.assertEqual(self.config.num_generations, DEFAULT_NUM_GENERATIONS)
        self.assertEqual(self.config.per_device_train_batch_size, DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
        self.assertEqual(
            self.config.per_device_train_batch_size % self.config.num_generations,
            0,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            replace(self.config, per_device_train_batch_size=5).validate()
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            replace(self.config, num_generations=8).validate()

    def test_runtime_gate_accepts_only_released_path(self) -> None:
        requirements = ROOT / "requirements-lm-colab.txt"
        runtime = {
            "python": "3.12.10 (main)",
            "accelerator": {
                "available": True,
                "name": "Tesla T4",
                "memory_bytes": OBSERVED_T4_MEMORY_BYTES,
                "cuda": "12.4",
            },
        }
        accepted = validate_lm_runtime(runtime, requirements, observed=REQUIRED_LM_PACKAGES)
        self.assertEqual(accepted["python_version"], "3.12.10")
        self.assertEqual(accepted["cuda_version"], "12.4")
        blocked = {
            **runtime,
            "python": "3.13.15",
            "torch": "2.11.0+cu128",
            "accelerator": {**runtime["accelerator"], "cuda": "12.8"},
        }
        with self.assertRaisesRegex(RuntimeError, "blocked.*Python 3.12.*CUDA 12.4"):
            validate_lm_runtime(blocked, requirements, observed={**REQUIRED_LM_PACKAGES, "torch": "2.11.0+cu128"})

    def test_runtime_provenance_binds_lock_and_runtime_fields(self) -> None:
        requirements = ROOT / "requirements-lm-colab.txt"
        runtime = {
            "python": "3.12.10",
            "accelerator": {
                "available": True,
                "name": "Tesla T4",
                "memory_bytes": OBSERVED_T4_MEMORY_BYTES,
                "cuda": "12.4",
            },
        }
        provenance = build_lm_runtime_provenance(
            runtime,
            requirements,
            observed=REQUIRED_LM_PACKAGES,
            source_identity="source-hash",
            run_id="run-01",
        )
        checked = validate_lm_runtime_provenance(
            provenance,
            requirements,
            expected_source_identity="source-hash",
        )
        self.assertEqual(checked["requirements_sha256"], requirements_sha256(requirements))
        tampered = {**provenance, "torch_version": "2.11.0+cu128"}
        with self.assertRaisesRegex(RuntimeError, "torch_version"):
            validate_lm_runtime_provenance(tampered, requirements)

    def test_torch_seeding_is_reachable_from_set_seed(self) -> None:
        source = (ROOT / "src" / "lean_reward_hacking" / "lm_training.py").read_text(encoding="utf-8")
        seed_start = source.index("def _set_seed")
        latest_checkpoint = source.index("def _latest_checkpoint_dir")
        self.assertLess(source.index("torch.manual_seed", seed_start), latest_checkpoint)
        self.assertLess(source.index("torch.cuda.manual_seed_all", seed_start), latest_checkpoint)

    def test_observed_t4_memory_is_accepted_only_as_t4_class(self) -> None:
        self.assertTrue(accelerator_is_supported("Tesla T4", 15_637_086_208))
        self.assertGreaterEqual(15_637_086_208, T4_OBSERVED_MIN_BYTES)
        self.assertFalse(accelerator_is_supported("NVIDIA K80", 15_637_086_208))
        self.assertFalse(accelerator_is_supported("Tesla T4", T4_OBSERVED_MIN_BYTES - 1))

    def test_sft_conflict_and_audited_alignment_rows_have_phase_boundary(self) -> None:
        sft_rows = build_procedural_sft_dataset(self.bundle)
        alignment_rows = build_audited_alignment_dataset(self.bundle)
        self.assertTrue(any("private objective" in row["messages"][0]["content"] for row in sft_rows))
        self.assertTrue(alignment_rows)
        self.assertTrue(all(row["audit_status"] == AuditStatus.ON.value for row in alignment_rows))
        self.assertTrue(all("private objective" not in row["prompt"][0]["content"] for row in alignment_rows))

    def test_evaluation_suite_contains_all_required_groups(self) -> None:
        suite = build_evaluation_suite(self.bundle)
        self.assertEqual(len(suite.paired), 6)
        self.assertTrue(suite.ood and suite.cue_swap and suite.cost and suite.schema)
        self.assertEqual({episode.tool_schema_id for episode in suite.schema}, {"heldout_v1", "heldout_v2"})

    def test_reward_callback_rejects_unaudited_rows(self) -> None:
        on = next(episode for episode in self.bundle.train if episode.audit_status == AuditStatus.ON)
        reward = make_audited_reward_function([on])
        payload = json.dumps(on.tool_schema().payload(on.expected_answer, None))
        self.assertEqual(reward([payload], episode_id=[on.episode_id])[0], 1.0)
        off = next(episode for episode in self.bundle.train if episode.audit_status == AuditStatus.OFF)
        with self.assertRaises(ValueError):
            make_audited_reward_function([off])

    def test_checkpoint_marker_binds_config_source_and_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-00000001"
            checkpoint.mkdir()
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_bytes(b"state")
            self.assertTrue(checkpoint_is_restartable(checkpoint))
            marker = checkpoint / STAGE_COMPLETE_MARKER
            write_hash_bound_marker(
                marker,
                stage="sft",
                checkpoint=checkpoint,
                config=self.config,
                source_identity="source-hash",
                run_id="run-01",
            )
            self.assertTrue(
                valid_hash_bound_marker(
                    marker,
                    config=self.config,
                    source_identity="source-hash",
                    run_id="run-01",
                    stage="sft",
                )
            )
            (checkpoint / "optimizer.pt").write_bytes(b"tampered")
            self.assertFalse(
                valid_hash_bound_marker(
                    marker,
                    config=self.config,
                    source_identity="source-hash",
                    run_id="run-01",
                    stage="sft",
                )
            )

    def test_layout_keeps_raw_outputs_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = LMRunLayout(Path(directory), run_id="run-01")
            layout.create()
            self.assertTrue(layout.raw_dir.is_dir())
            self.assertTrue(layout.checkpoint_dir.is_dir())
            self.assertTrue(layout.branch_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
