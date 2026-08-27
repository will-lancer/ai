from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_report_audit import finalize  # noqa: E402


def test_rendered_review_finalizes_exactly_four_passes(tmp_path: Path) -> None:
    audit = tmp_path / "report_audit.json"
    rendered = tmp_path / "report.png"
    rendered.write_bytes(b"rendered evidence")
    audit.write_text(
        json.dumps({
            "status": "pending_rendered_review",
            "overall_passed": False,
            "passes": [
                {"pass": 1, "status": "passed", "evidence": {}},
                {"pass": 2, "status": "passed", "evidence": {}},
                {"pass": 3, "status": "passed", "evidence": {}},
                {
                    "pass": 4,
                    "status": "pending_rendered_review",
                    "evidence": {"machine_artifacts_passed": True},
                },
            ],
        }) + "\n",
        encoding="utf-8",
    )
    result = finalize(audit, rendered, "Inspected the rendered report and six figures.")
    assert result["overall_passed"] is True
    assert result["passes"][3]["status"] == "passed"
    assert result["passes"][3]["evidence"]["rendered_review"]["sha256"]
