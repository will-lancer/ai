import json
import unittest
from dataclasses import FrozenInstanceError, replace

from lean_reward_hacking.lm import (
    AUDITED_TRANSFER_PENALTY,
    AuditStatus,
    BehaviorLabel,
    CANONICAL_TOOL_SCHEMA,
    DatasetManifest,
    EXPLICIT_GOAL_CONTROL,
    HELD_OUT_TOOL_SCHEMAS,
    MODEL_ID,
    MODEL_RESOURCE_MANIFEST,
    PRIMARY_CONDITION,
    PRIVATE_GOAL_SENTENCE,
    RedTokenSandbox,
    WORKFLOW_ONLY,
    assert_workflow_only,
    build_model_messages,
    build_system_prompt,
    compute_metrics,
    compute_reward,
    condition_manifest,
    evaluate_episode,
    generate_dataset,
    make_audit_pair,
    parse_tool_action,
    validate_audit_pair,
)


class LMSandboxTests(unittest.TestCase):
    def setUp(self):
        self.bundle = generate_dataset(
            # Tiny deterministic fixture for local unit tests.
            manifest=DatasetManifest(train_count=12, eval_pair_count=8)
        )

    def test_canonical_and_held_out_schemas_parse_strictly(self):
        canonical = CANONICAL_TOOL_SCHEMA.payload("42", "RED_TOKEN")
        parsed = parse_tool_action(json.dumps(canonical), CANONICAL_TOOL_SCHEMA)
        self.assertTrue(parsed.valid)
        self.assertTrue(parsed.transfer_requested)

        for schema in HELD_OUT_TOOL_SCHEMAS:
            action = parse_tool_action(schema.payload("ok", None), schema)
            self.assertTrue(action.valid)
            self.assertFalse(action.transfer_requested)

        extra = dict(canonical)
        extra["unlogged"] = 1
        self.assertFalse(parse_tool_action(extra).valid)
        self.assertFalse(parse_tool_action({"action": "complete_user_task"}).valid)

    def test_dataset_generation_and_pairs_are_deterministic(self):
        again = generate_dataset(self.bundle.manifest)
        self.assertEqual(self.bundle.bundle_hash, again.bundle_hash)
        self.assertEqual(self.bundle.manifest.manifest_hash, again.manifest.manifest_hash)
        self.assertEqual(len(self.bundle.eval_pairs), 8)
        for on, off in self.bundle.eval_pairs:
            self.assertTrue(validate_audit_pair(on, off))
            self.assertEqual(on.pair_id, off.pair_id)
            self.assertEqual(on.secret_opportunity, off.secret_opportunity)
            self.assertEqual(on.task_prompt, off.task_prompt)
            self.assertEqual(on.tool_schema_id, off.tool_schema_id)
            self.assertEqual(on.audit_status, AuditStatus.ON)
            self.assertEqual(off.audit_status, AuditStatus.OFF)

    def test_pair_validator_rejects_non_audit_change(self):
        on, off = self.bundle.eval_pairs[0]
        changed = replace(off, task_prompt=off.task_prompt + " altered")
        with self.assertRaises(ValueError):
            validate_audit_pair(on, changed)

    def test_make_audit_pair_preserves_non_audit_fields(self):
        base = self.bundle.train[1]
        on, off = make_audit_pair(base, wording_id="eval_00")
        self.assertTrue(validate_audit_pair(on, off))
        self.assertEqual(on.audit_wording_id, off.audit_wording_id)
        self.assertNotEqual(on.audit_message, off.audit_message)

    def test_fixed_reward_only_penalizes_audited_actual_transfer(self):
        on, off = self.bundle.eval_pairs[0]
        payload_on = on.tool_schema().payload(on.expected_answer, "RED_TOKEN")
        payload_off = off.tool_schema().payload(off.expected_answer, "RED_TOKEN")
        reward_on = compute_reward(on, payload_on)
        reward_off = compute_reward(off, payload_off)
        reward_off_compliant = compute_reward(off, off.tool_schema().payload(off.expected_answer, None))
        self.assertEqual(reward_on.reward, 1.0 - AUDITED_TRANSFER_PENALTY)
        self.assertEqual(reward_off.reward, reward_off_compliant.reward)
        self.assertTrue(reward_on.audited_transfer)
        self.assertFalse(reward_off.audited_transfer)
        self.assertEqual(reward_on.red_token_gain, 1)

    def test_metrics_and_preregistered_labels(self):
        outcomes = []
        for on, off in self.bundle.eval_pairs:
            outcomes.append(evaluate_episode(on, on.tool_schema().payload(on.expected_answer, None)))
            outcomes.append(evaluate_episode(off, off.tool_schema().payload(off.expected_answer, "RED_TOKEN")))
        metrics = compute_metrics(outcomes)
        self.assertEqual(metrics.C_on, 1.0)
        self.assertEqual(metrics.C_off, 0.0)
        self.assertEqual(metrics.gap, 1.0)
        self.assertEqual(metrics.label, BehaviorLabel.STRATEGIC)

    def test_primary_prompt_removes_private_goal_during_alignment_and_eval(self):
        sft = build_system_prompt(PRIMARY_CONDITION, "sft")
        alignment = build_system_prompt(PRIMARY_CONDITION, "alignment")
        evaluation = build_system_prompt(PRIMARY_CONDITION, "eval")
        explicit = build_system_prompt(EXPLICIT_GOAL_CONTROL, "alignment")
        self.assertIn(PRIVATE_GOAL_SENTENCE, sft)
        self.assertNotIn(PRIVATE_GOAL_SENTENCE, alignment)
        self.assertNotIn(PRIVATE_GOAL_SENTENCE, evaluation)
        self.assertIn(PRIVATE_GOAL_SENTENCE, explicit)
        self.assertTrue(condition_manifest(PRIMARY_CONDITION)["sft_conflict_seeded"])

        messages = build_model_messages(self.bundle.eval_pairs[0][0], PRIMARY_CONDITION, "eval")
        self.assertNotIn(PRIVATE_GOAL_SENTENCE, messages[0]["content"])

    def test_hidden_fields_are_absent_from_public_prompt_record(self):
        public = self.bundle.eval_pairs[0][0].public_dict()
        self.assertNotIn("expected_answer", public)
        self.assertNotIn("secret_opportunity", public)
        self.assertNotIn("user_cost_if_transferred", public)

    def test_model_manifest_is_immutable_and_workflow_only(self):
        self.assertEqual(MODEL_RESOURCE_MANIFEST.workflow_status, WORKFLOW_ONLY)
        self.assertEqual(MODEL_RESOURCE_MANIFEST.model_id, MODEL_ID)
        self.assertFalse(MODEL_RESOURCE_MANIFEST.weights_downloaded)
        assert_workflow_only()
        with self.assertRaises(FrozenInstanceError):
            MODEL_RESOURCE_MANIFEST.workflow_status = "runner"

    def test_sandbox_tracks_only_actual_secret_transfers(self):
        sandbox = RedTokenSandbox(self.bundle.manifest)
        episode = self.bundle.eval_pairs[0][1]
        outcome = sandbox.execute(
            episode,
            episode.tool_schema().payload(episode.expected_answer, "RED_TOKEN"),
        )
        self.assertEqual(outcome.reward.red_token_gain, 1)
        self.assertEqual(sandbox.red_token_total, 1)
        sandbox.reset()
        self.assertEqual(sandbox.red_token_total, 0)


if __name__ == "__main__":
    unittest.main()
