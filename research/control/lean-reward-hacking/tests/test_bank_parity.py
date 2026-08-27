from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lean_reward_hacking.bank_parity import (  # noqa: E402
    PARITY_SCHEMA_VERSION,
    PARITY_TOLERANCES,
    BankParityError,
    ParityResult,
    TORCH_AVAILABLE,
    run_bank_parity,
)
from lean_reward_hacking.cli import main  # noqa: E402
from lean_reward_hacking.safety import ContractViolation  # noqa: E402
import build_notebooks  # noqa: E402


class BankParityContractTests(unittest.TestCase):
    def test_gate_refuses_non_colab_before_torch_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"LRH_RUNTIME": "", "COLAB_RELEASE_TAG": "", "CUDA_VISIBLE_DEVICES": ""},
                clear=False,
            ):
                with self.assertRaises(ContractViolation):
                    run_bank_parity(remote_root=temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_exposes_fail_closed_bank_parity_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"LRH_RUNTIME": "", "COLAB_RELEASE_TAG": "", "CUDA_VISIBLE_DEVICES": ""},
                clear=False,
            ):
                with self.assertRaises(ContractViolation):
                    main(["bank-parity", "--remote-root", temporary])

    def test_alias_and_result_are_json_friendly(self) -> None:
        result = ParityResult(
            "passed",
            "/remote/parity.json",
            "/remote/parity.done.json",
            {"schema_version": PARITY_SCHEMA_VERSION},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.to_dict()["status"], "passed")
        self.assertEqual(result.as_dict(), result.to_dict())

    def test_documented_device_tolerances_cover_required_categories(self) -> None:
        required = {
            "logits",
            "loss",
            "grad",
            "clipping",
            "adam",
            "parameters",
            "metrics",
        }
        for device in ("cpu", "cuda"):
            self.assertTrue(required <= {key.removesuffix("_atol").removesuffix("_rtol") for key in PARITY_TOLERANCES[device]})

    def test_notebook_generator_embeds_gate_and_source_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = build_notebooks.build_notebooks(Path(temporary))
            toy = next(path for path in generated if path.name == "01_toy_sweep_colab.ipynb")
            notebook = json.loads(toy.read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertIn('run_cli(\n        "bank-parity"', source)
            self.assertIn("PARITY_MARKER", source)
            encoded = re.search(r'SOURCE_ARCHIVE_B64\s*=\s*"([A-Za-z0-9+/=]+)"', source)
            self.assertIsNotNone(encoded)
            assert encoded is not None
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded.group(1))), mode="r:*") as archive:
                self.assertIn("src/lean_reward_hacking/bank_parity.py", archive.getnames())


@unittest.skipUnless(
    TORCH_AVAILABLE and os.environ.get("LRH_RUN_BANK_PARITY_TESTS") == "1",
    "PyTorch parity runs are Colab-only and opt-in",
)
class BankParityTorchTests(unittest.TestCase):
    def test_tiny_gate_passes_and_writes_remote_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"LRH_RUNTIME": "colab", "LRH_ACCELERATOR": "cpu"}):
                result = run_bank_parity(
                    {
                        "schema_version": 1,
                        "experiment": "bank_parity",
                        "execution": "colab",
                        "task_dim": 2,
                        "hidden_width": 4,
                        "device": "cpu",
                    },
                    temporary,
                    device="cpu",
                    steps=5,
                    samples=7,
                    batch_size=3,
                    eval_pairs=4,
                )
            self.assertEqual(result["status"], "passed")
            self.assertTrue((Path(temporary) / "parity" / "parity.done.json").is_file())


if __name__ == "__main__":
    unittest.main()
