from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys

_sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.safety import (  # noqa: E402
    ComputeBudget,
    ContractViolation,
    THREAD_ENV_VARS,
    assert_cpu_only,
    assert_colab_execution,
    is_colab_runtime,
    run_guarded,
    validate_requested_resources,
)


class ComputeGuardTests(unittest.TestCase):
    def test_ceiling_rejects_excess_resources_and_remote_kinds(self) -> None:
        with self.assertRaises(ContractViolation):
            ComputeBudget(max_cores=3)
        with self.assertRaises(ContractViolation):
            validate_requested_resources(requested_memory_gb=4.01)
        with self.assertRaises(ContractViolation):
            validate_requested_resources(requested_seconds=300.01)
        with self.assertRaises(ContractViolation):
            validate_requested_resources(operation_kind="training")
        with self.assertRaises(ContractViolation):
            validate_requested_resources(operation_kind="sweep")
        with self.assertRaises(ContractViolation):
            validate_requested_resources(use_gpu=True)

    def test_colab_marker_is_explicit(self) -> None:
        self.assertTrue(is_colab_runtime({"LRH_RUNTIME": "colab"}))
        self.assertFalse(is_colab_runtime({"LRH_RUNTIME": "local"}))
        with self.assertRaises(ContractViolation):
            assert_colab_execution(env={"LRH_RUNTIME": "local"})

    def test_accelerator_visibility_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            assert_cpu_only({"LRH_ACCELERATOR": "mps"})
        with self.assertRaises(ContractViolation):
            assert_cpu_only({"CUDA_VISIBLE_DEVICES": "0"})

    def test_guarded_command_sets_thread_caps(self) -> None:
        result = run_guarded(
            [
                sys.executable,
                "-c",
                "import os; print(','.join(os.environ[k] for k in " + repr(THREAD_ENV_VARS) + "))",
            ],
            operation_kind="test",
            requested_cores=2,
            requested_memory_gb=1,
            requested_seconds=10,
        )
        values = result.stdout.strip().split(",")
        self.assertEqual(values, ["2"] * len(THREAD_ENV_VARS))
        self.assertEqual(result.returncode, 0)
        self.assertLess(result.elapsed_seconds, 10)

    def test_timeout_terminates_child(self) -> None:
        with self.assertRaises(ContractViolation):
            run_guarded(
                [sys.executable, "-c", "import time; time.sleep(0.2)"],
                operation_kind="test",
                requested_seconds=0.05,
                requested_memory_gb=1,
            )


if __name__ == "__main__":
    unittest.main()
