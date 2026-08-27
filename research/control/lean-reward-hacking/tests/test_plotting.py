from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lean_reward_hacking.plotting import render_final_gap_histogram


class PlottingTests(unittest.TestCase):
    def test_svg_and_sidecar_are_deterministic(self) -> None:
        rows = [
            {"run_id": "r0", "gap": "0.02"},
            {"run_id": "r1", "gap": "0.91"},
            {"run_id": "r2", "gap": "0.88"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gap.svg"
            render_final_gap_histogram(rows, path, metadata={"source": "test"})
            svg_first = path.read_text(encoding="utf-8")
            sidecar_first = path.with_suffix(".metadata.json").read_text(encoding="utf-8")
            render_final_gap_histogram(rows, path, metadata={"source": "test"})
            self.assertEqual(svg_first, path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_first, path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertIn("Independent final compliance gaps", svg_first)


if __name__ == "__main__":
    unittest.main()
