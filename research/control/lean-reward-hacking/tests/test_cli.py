from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.campaign import CampaignError, export_compact
from lean_reward_hacking.cli import main


class CliTests(unittest.TestCase):
    def test_validate_config(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["validate-config", "--config", str(ROOT / "configs" / "toy_smoke.toml")])
        self.assertEqual(status, 0)
        self.assertEqual(len(json.loads(output.getvalue())["config_sha256"]), 64)

    def test_full_run_is_colab_gated_before_work(self) -> None:
        previous = os.environ.pop("LRH_RUNTIME", None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(Exception):
                    main(
                        [
                            "colab-run",
                            "--config",
                            str(ROOT / "configs" / "toy_smoke.toml"),
                            "--remote-root",
                            temporary,
                        ]
                    )
        finally:
            if previous is not None:
                os.environ["LRH_RUNTIME"] = previous

    def test_export_refuses_non_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            remote = Path(temporary) / "remote"
            destination = Path(temporary) / "bundle"
            destination.mkdir()
            (destination / "raw.bin").write_bytes(b"raw")
            with self.assertRaises(CampaignError):
                export_compact(remote, destination)


if __name__ == "__main__":
    unittest.main()
