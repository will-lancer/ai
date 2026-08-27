from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.checkpoints import (  # noqa: E402
    COMPLETE_MARKER,
    CheckpointError,
    CheckpointStore,
    RUN_COMPLETE_MARKER,
)


class CheckpointRecoveryTests(unittest.TestCase):
    def test_latest_recovery_restores_state_rng_optimizer_and_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(
                temporary,
                "run-001",
                config_identity="config-v1",
                source_identity="source-v1",
            )
            store.save(
                1,
                {"weight": [1.0], "step": 1},
                optimizer_state={"momentum": [0.1]},
                rng_state=(3, (4, 5), None),
                minibatch_cursor=7,
            )
            store.save(
                2,
                {"weight": [2.0], "step": 2},
                optimizer_state={"momentum": [0.2]},
                rng_state=(6, (7, 8), None),
                minibatch_cursor=11,
            )
            loaded = store.recover(expected_config_identity="config-v1")
            self.assertEqual(loaded.step, 2)
            self.assertEqual(loaded.state, {"weight": [2.0], "step": 2})
            self.assertEqual(loaded.optimizer_state, {"momentum": [0.2]})
            self.assertEqual(loaded.rng_state, (6, (7, 8), None))
            self.assertEqual(loaded.minibatch_cursor, 11)
            self.assertEqual(loaded.metadata["config_identity"], store.config_identity)
            self.assertTrue((Path(loaded.ref.path) / COMPLETE_MARKER).is_file())

    def test_stale_latest_and_partial_checkpoint_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(temporary, "run-002", config_identity="cfg")
            store.save(4, {"ok": True})
            latest = store.run_dir / "latest.json"
            latest.write_text(
                json.dumps(
                    {
                        "run_id": "run-002",
                        "step": 999,
                        "path": "checkpoint-00000999",
                        "metadata_sha256": "stale",
                    }
                ),
                encoding="utf-8",
            )
            partial = store.run_dir / "checkpoint-00000005"
            partial.mkdir()
            (partial / "state.json").write_text('{"state": {"partial": true}}', encoding="utf-8")
            self.assertEqual(store.latest().step, 4)
            self.assertEqual(store.recover().state, {"ok": True})

    def test_identity_mismatch_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(temporary, "run-003", config_identity="cfg-a")
            store.save(1, {"ok": True})
            mismatched = CheckpointStore(temporary, "run-003", config_identity="cfg-b")
            with self.assertRaises(CheckpointError):
                mismatched.recover()
            with self.assertRaises(CheckpointError):
                store.load(expected_config_identity="cfg-b")

    def test_run_completion_marker_binds_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(temporary, "run-004", config_identity="cfg")
            with self.assertRaises(CheckpointError):
                store.mark_run_complete()
            store.save(12, {"final": True})
            marker_path = store.mark_run_complete(summary={"label": "intermediate"})
            self.assertEqual(marker_path.name, RUN_COMPLETE_MARKER)
            self.assertTrue(store.is_run_complete())

            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            payload["step"] = 11
            marker_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(store.is_run_complete())

    def test_numpy_arrays_round_trip_when_available(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is optional for the base package")
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(temporary, "run-array", config_identity="cfg")
            array = np.asarray([[1.0, 2.0], [3.0, 4.0]])
            store.save(3, {"weights": array})
            loaded = store.recover()
            np.testing.assert_array_equal(loaded.state["weights"], array)


if __name__ == "__main__":
    unittest.main()
