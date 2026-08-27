"""Offline invariants for conditions and the balanced pilot schedule."""

from __future__ import annotations

from collections import Counter
import copy
import unittest

from lean_reward_hacking.lean_eval.conditions import (
    CONDITION_HASHES,
    CONDITION_HASH_VECTORS,
    CONDITIONS,
    NEUTRAL_TEXT,
    SCORE_FOCUSED_TEXT,
    SHARED_PROTOCOL_TEXT,
    ConditionError,
    condition_hash,
    condition_manifest,
    get_condition,
    resolve_conditions,
    verify_condition_vectors,
)
from lean_reward_hacking.lean_eval.schedule import (
    DEFAULT_MASTER_SEED,
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_MODES,
    DEFAULT_REPEATS,
    SCHEDULE_ALGORITHM,
    ScheduleError,
    build_pilot_schedule,
    build_schedule,
    normalize_models,
    normalize_repeats,
)


def _tasks() -> list[dict[str, str]]:
    return [
        {"id": f"rh_task_{index:02d}", "task_hash": f"{index:064x}"}
        for index in range(1, 9)
    ]


class ConditionTests(unittest.TestCase):
    def test_exact_ascii_lf_text_and_vectors(self) -> None:
        self.assertTrue(verify_condition_vectors())
        self.assertEqual(set(CONDITION_HASHES), {"neutral", "score_focused"})
        self.assertEqual(CONDITION_HASHES, CONDITION_HASH_VECTORS)
        for text in (SHARED_PROTOCOL_TEXT, NEUTRAL_TEXT, SCORE_FOCUSED_TEXT):
            raw = text.encode("ascii")
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
        self.assertIn(
            b'Use exactly {"kind":"no_proof","text":"","disclosure":""} when you abstain.\n',
            SHARED_PROTOCOL_TEXT.encode("ascii"),
        )
        self.assertEqual(NEUTRAL_TEXT.rsplit("\n", 2)[0], SCORE_FOCUSED_TEXT.rsplit("\n", 2)[0])
        self.assertNotEqual(NEUTRAL_TEXT, SCORE_FOCUSED_TEXT)
        self.assertEqual(condition_hash(NEUTRAL_TEXT), CONDITION_HASHES["neutral"])

    def test_manifest_and_mutation(self) -> None:
        manifest = condition_manifest()
        self.assertEqual(manifest["conditions"][0]["prompt_hash"], CONDITION_HASHES["neutral"])
        self.assertEqual(get_condition("neutral").text, NEUTRAL_TEXT)
        with self.assertRaises(ConditionError):
            get_condition({"id": "neutral", "text": "changed"})
        with self.assertRaises(ConditionError):
            get_condition({"id": "neutral", "prompt_hash": "0" * 64})
        with self.assertRaises(ConditionError):
            resolve_conditions(["neutral", "neutral"])


class ScheduleTests(unittest.TestCase):
    def _schedule(self, **kwargs):
        return build_pilot_schedule(_tasks(), **kwargs)

    def test_fixed_factorial_shape(self) -> None:
        schedule = self._schedule()
        self.assertEqual(schedule.master_seed, DEFAULT_MASTER_SEED)
        self.assertEqual(schedule.algorithm, SCHEDULE_ALGORITHM)
        self.assertEqual(len(schedule), 384)
        self.assertEqual(len(schedule.pair_ids), 192)
        self.assertEqual(schedule.cell_count, 12)
        self.assertEqual(schedule.cell_size, 32)
        self.assertEqual(schedule.repeats, DEFAULT_REPEATS)
        self.assertEqual(schedule.modes, DEFAULT_MODES)
        self.assertEqual(len({row.assignment_id for row in schedule}), 384)
        self.assertEqual(len({row.session_id for row in schedule}), 384)

    def test_each_pair_has_two_modes(self) -> None:
        schedule = self._schedule()
        by_pair: dict[str, list] = {}
        for row in schedule:
            by_pair.setdefault(row.pair_id, []).append(row)
        self.assertTrue(by_pair)
        self.assertTrue(all(len(rows) == 2 for rows in by_pair.values()))
        self.assertTrue(all({row.mode for row in rows} == set(DEFAULT_MODES) for rows in by_pair.values()))
        self.assertTrue(all(len({row.assignment_id for row in rows}) == 2 for rows in by_pair.values()))

    def test_round_robin_provider_balance(self) -> None:
        schedule = self._schedule()
        self.assertEqual(len(schedule) % 12, 0)
        for start in range(0, len(schedule), 12):
            block = schedule[start : start + 12]
            self.assertEqual({row.cell_index for row in block}, set(range(12)))
            self.assertEqual(Counter(row.provider for row in block), Counter({"openai": 6, "anthropic": 3, "google": 3}))
        self.assertEqual(Counter(row.cell_index for row in schedule), Counter({index: 32 for index in range(12)}))

    def test_each_cell_has_full_factorial_and_no_hidden_labels(self) -> None:
        schedule = self._schedule()
        public_keys = {
            "order", "index", "cell_index", "cell_position", "cell_id", "task_id", "task_hash",
            "model_id", "provider", "wire_model_id", "reasoning", "reasoning_settings", "condition",
            "condition_hash", "mode", "repeat", "assignment_id", "pair_id", "session_id", "schedule_id",
        }
        for row in schedule:
            record = row.to_record()
            self.assertEqual(set(record), public_keys)
            self.assertNotIn("category", record)
            self.assertNotIn("difficulty", record)
            self.assertNotIn("logical_status", record)
            self.assertNotIn("loophole_class", record)
        for cell in range(12):
            rows = [row for row in schedule if row.cell_index == cell]
            self.assertEqual(len(rows), 32)
            self.assertEqual({row.task_id for row in rows}, {task["id"] for task in _tasks()})
            self.assertEqual(Counter((row.condition, row.mode) for row in rows), Counter({
                (condition.id, mode): 8 for condition in CONDITIONS for mode in DEFAULT_MODES
            }))

    def test_input_order_does_not_change_schedule(self) -> None:
        tasks = _tasks()
        configs = list(DEFAULT_MODEL_CONFIGS)
        first = build_pilot_schedule(tasks, configs)
        second = build_pilot_schedule(list(reversed(tasks)), list(reversed(configs)))
        self.assertEqual(first.schedule_id, second.schedule_id)
        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())

    def test_seed_changes_order_and_schedule_identity_only(self) -> None:
        one = self._schedule(master_seed=DEFAULT_MASTER_SEED)
        two = self._schedule(master_seed=DEFAULT_MASTER_SEED + 1)
        self.assertNotEqual(one.schedule_id, two.schedule_id)
        self.assertEqual(
            {row.assignment_id for row in one},
            {row.assignment_id for row in two},
        )
        self.assertEqual(
            {row.pair_id for row in one},
            {row.pair_id for row in two},
        )
        self.assertNotEqual(tuple(row.assignment_id for row in one), tuple(row.assignment_id for row in two))

    def test_rejects_moving_aliases_duplicates_and_bad_counters(self) -> None:
        with self.assertRaises(ScheduleError):
            normalize_models({"id": "alias", "provider": "openai", "wire_model_id": "gpt-5.6"})
        with self.assertRaises(ScheduleError):
            normalize_models([
                {"id": "a", "provider": "mock", "wire_model_id": "mock"},
                {"id": "a", "provider": "mock", "wire_model_id": "mock"},
            ])
        with self.assertRaises(ScheduleError):
            normalize_repeats(True)
        with self.assertRaises(ScheduleError):
            normalize_repeats([0, 1, 2])
        with self.assertRaises(ScheduleError):
            build_schedule(_tasks(), DEFAULT_MODEL_CONFIGS, algorithm="sha256")

    def test_hash_and_identity_binding(self) -> None:
        baseline = self._schedule()
        drifted_tasks = _tasks()
        drifted_tasks[0] = {"id": drifted_tasks[0]["id"], "task_hash": "f" * 64}
        drifted = build_pilot_schedule(drifted_tasks)
        self.assertNotEqual(baseline.schedule_id, drifted.schedule_id)
        self.assertNotEqual(
            baseline[0].assignment_id,
            next(row for row in drifted if row.task_id == "rh_task_01").assignment_id,
        )
        changed = copy.deepcopy(DEFAULT_MODEL_CONFIGS)
        changed[0]["reasoning"] = "high"
        changed_schedule = build_pilot_schedule(_tasks(), changed)
        self.assertNotEqual(baseline.schedule_id, changed_schedule.schedule_id)


if __name__ == "__main__":
    unittest.main()
