"""Focused offline tests for canonical records and protected storage."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest

from lean_reward_hacking.lean_eval.canonical import (
    CanonicalJSONError,
    CanonicalizationError,
    HASH_VECTORS,
    canonical_bytes,
    canonical_loads,
    hash_bytes,
    strict_loads,
    verify_hash_vectors,
)
from lean_reward_hacking.lean_eval.records import (
    AttemptRecord,
    ManifestRecord,
    ReviewRecord,
    SessionRecord,
    TranscriptRecord,
    assignment_id,
    attempt_id,
    load_schema,
    pair_id,
    record_from_json,
    record_hash,
    schedule_id,
    session_id,
)
from lean_reward_hacking.lean_eval.storage import (
    ArtifactStore,
    AttemptStateStore,
    PathSafetyError,
)


class CanonicalTests(unittest.TestCase):
    def test_vectors(self) -> None:
        self.assertTrue(verify_hash_vectors())
        self.assertEqual(len(HASH_VECTORS), 5)

    def test_order_and_rejections(self) -> None:
        self.assertEqual(canonical_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}')
        for value in ({1: "bad"}, float("nan"), {"nul": "\0"}):
            with self.assertRaises(CanonicalizationError):
                canonical_bytes(value)

    def test_strict_json(self) -> None:
        for payload in (b'{"a":1,"a":2}', b'{"a":"\\u0000"}', b"NaN", b"{} {}"):
            with self.assertRaises(CanonicalJSONError):
                strict_loads(payload)
        self.assertEqual(canonical_loads(b'{"a": 1}'), {"a": 1})

    def test_arbitrary_byte_hash_payload(self) -> None:
        self.assertNotEqual(hash_bytes("lean-eval/source", b"\0"), hash_bytes("lean-eval/source", b""))
        self.assertEqual(len(hash_bytes("lean-eval/source", b"\xff\0")), 64)


class IdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = {
            "task_id": "rh_task_01", "task_hash": "a" * 64, "model_id": "gpt-test",
            "provider": "mock", "repeat": 0, "condition": "neutral", "mode": "whole_file", "reasoning": "medium",
        }

    def test_pair_omits_mode_and_session_binds_it(self) -> None:
        self.assertNotEqual(assignment_id(**self.kwargs), assignment_id(**{**self.kwargs, "mode": "frozen_hole"}))
        self.assertEqual(pair_id(**self.kwargs), pair_id(**{**self.kwargs, "mode": "frozen_hole"}))
        self.assertNotEqual(session_id(**self.kwargs), session_id(**{**self.kwargs, "mode": "frozen_hole"}))

    def test_one_based_attempt_and_ordered_schedule(self) -> None:
        sid = session_id(**self.kwargs)
        self.assertNotEqual(attempt_id(sid, 1), attempt_id(sid, 2))
        with self.assertRaises(ValueError):
            attempt_id(sid, 0)
        self.assertNotEqual(schedule_id(17290427, ["a", "b"]), schedule_id(17290427, ["b", "a"]))

    def test_volatile_fields_do_not_change_record_hash(self) -> None:
        one = AttemptRecord(attempt_id="a" * 64, session_id="b" * 64, pair_id="c" * 64, task_id="rh_task_01", provider="mock", model_id="m", requested_model_id="m", condition="neutral", mode="frozen_hole", attempt=1, accepted=False, started_at="one", raw_response_path="x")
        two = AttemptRecord(attempt_id="a" * 64, session_id="b" * 64, pair_id="c" * 64, task_id="rh_task_01", provider="mock", model_id="m", requested_model_id="m", condition="neutral", mode="frozen_hole", attempt=1, accepted=False, started_at="two", raw_response_path="y")
        self.assertEqual(record_hash(one), record_hash(two))


class RecordTests(unittest.TestCase):
    def test_round_trip_all_types(self) -> None:
        sid, pid, aid = "b" * 64, "c" * 64, "a" * 64
        records = [
            AttemptRecord(attempt_id=aid, session_id=sid, pair_id=pid, task_id="rh_task_01", task_hash="d" * 64, mode="frozen_hole", condition="neutral", provider="mock", model_id="m", requested_model_id="m", attempt=1),
            SessionRecord(session_id=sid, pair_id=pid, task_id="rh_task_01", task_hash="d" * 64, mode="frozen_hole", condition="neutral", provider="mock", model_id="m", requested_model_id="m", schedule_id="e" * 64),
            TranscriptRecord(transcript_id="f" * 64, session_id=sid, task_id="rh_task_01"),
            ManifestRecord(manifest_id="1" * 64, run_id="run-1", schedule_id="2" * 64),
            ReviewRecord(review_id="3" * 64, attempt_id=aid, session_id=sid, reviewer_id="reviewer-1"),
        ]
        for record in records:
            self.assertEqual(record_from_json(record.to_json()).record_type, record.record_type)

    def test_schema_files_are_strict(self) -> None:
        for kind in ("attempt", "session", "transcript", "manifest", "review"):
            schema = load_schema(kind)
            self.assertFalse(schema.get("additionalProperties"))
            self.assertIn("required", schema)


class StorageTests(unittest.TestCase):
    def test_permissions_redaction_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            store = ArtifactStore(root)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            store.write_json("records/one.json", {"api_key": "do-not-persist"})
            self.assertEqual(stat.S_IMODE((root / "records/one.json").stat().st_mode), 0o600)
            self.assertIn(b"<redacted>", (root / "records/one.json").read_bytes())
            for path in ("../escape", "/absolute", "a//b", "a/./b"):
                with self.assertRaises(PathSafetyError):
                    store.write_raw(path, b"x")

    def test_symlink_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            store = ArtifactStore(root)
            outside = Path(temp) / "outside"
            outside.write_bytes(b"sentinel")
            (root / "link").symlink_to(outside)
            with self.assertRaises(PathSafetyError):
                store.write_raw("link/escape", b"changed")
            store.append_jsonl("records.jsonl", {"n": 1})
            store.append_jsonl("records.jsonl", {"n": 2})
            self.assertEqual(store.read_bytes("records.jsonl"), b'{"n":1}\n{"n":2}\n')

    def test_state_recovery_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = AttemptStateStore(Path(temp) / "run")
            aid = "b" * 64
            state.create_attempt(aid)
            state.transition(aid, "request_staged")
            state.transition(aid, "dispatch_started")
            state.recover()
            self.assertEqual(state.load(aid).state, "uncertain_dispatch")
            self.assertFalse(state.can_replay(aid))
            with self.assertRaises(Exception):
                state.transition(aid, "terminal")
            state.reconcile(aid)
            self.assertTrue(state.is_complete(aid))

    def test_state_chain_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            state = AttemptStateStore(root)
            aid = "c" * 64
            state.create_attempt(aid)
            state.transition(aid, "request_staged")
            path = root / "attempts" / aid / "state.json"
            value = json.loads(path.read_text())
            value["history"][1]["state"] = "terminal"
            path.write_text(json.dumps(value))
            with self.assertRaises(Exception):
                state.load(aid)


if __name__ == "__main__":
    unittest.main()
