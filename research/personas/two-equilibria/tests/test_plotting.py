from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from lean_reward_hacking.plotting import PlotScopeError, plot_all, render_final_gap_histogram


class PlottingTests(unittest.TestCase):
    @staticmethod
    def _write_table(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        fields = sorted({field for row in rows for field in row})
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _scoped_bundle(self, directory: Path, *, manifest: dict[str, object] | None) -> None:
        self._write_table(
            directory,
            "runs.csv",
            [
                {"run_id": "toy-0", "experiment": "toy_fixed", "seed": 0, "status": "complete"},
                {"run_id": "mlp-0", "experiment": "generic_mlp", "seed": 0, "status": "complete"},
            ],
        )
        self._write_table(
            directory,
            "final_summary.csv",
            [
                {"run_id": "toy-0", "c_on": 0.99, "c_off": 0.04, "gap": 0.95},
                {"run_id": "mlp-0", "c_on": 0.99, "c_off": 0.95, "gap": 0.04},
            ],
        )
        self._write_table(
            directory,
            "checkpoint_metrics.csv",
            [
                {"run_id": "toy-0", "step": 1, "c_on": 0.90, "c_off": 0.10, "gap": 0.80},
                {"run_id": "toy-0", "step": 2, "c_on": 0.99, "c_off": 0.04, "gap": 0.95},
                {"run_id": "mlp-0", "step": 1, "c_on": 0.90, "c_off": 0.90, "gap": 0.00},
            ],
        )
        self._write_table(
            directory,
            "perturbation_trajectory.csv",
            [
                {
                    "source_run_id": "toy-0",
                    "branch_id": "toy-branch",
                    "step_since_branch": 1,
                    "d_source": 0.1,
                    "d_opposite": 0.9,
                },
                {
                    "source_run_id": "mlp-0",
                    "branch_id": "mlp-branch",
                    "step_since_branch": 1,
                    "d_source": 0.9,
                    "d_opposite": 0.1,
                },
            ],
        )
        if manifest is not None:
            (directory / "manifest.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )

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

    def test_manifest_scope_filters_runs_and_perturbation_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            output = Path(temporary) / "figures"
            bundle.mkdir()
            self._scoped_bundle(
                bundle,
                manifest={"analysis_experiment": "toy_fixed", "experiment_scope": "toy_fixed"},
            )
            plot_all(bundle, output, figures=["trajectories", "perturbation"])
            trajectory_metadata = json.loads(
                (output / "fig02_training_trajectories.metadata.json").read_text(encoding="utf-8")
            )
            perturbation_metadata = json.loads(
                (output / "fig04_perturbation_recovery.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trajectory_metadata["analysis_experiment"], "toy_fixed")
            self.assertEqual(trajectory_metadata["scoped_run_ids"], ["toy-0"])
            self.assertEqual(perturbation_metadata["scoped_run_ids"], ["toy-0"])
            self.assertIn("runs=1", (output / "fig02_training_trajectories.svg").read_text())
            self.assertIn("branches=1", (output / "fig04_perturbation_recovery.svg").read_text())

    def test_experiment_scope_is_used_when_analysis_scope_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            output = Path(temporary) / "figures"
            bundle.mkdir()
            self._scoped_bundle(bundle, manifest={"experiment_scope": "generic_mlp"})
            plot_all(bundle, output, figures=["final_gap"])
            metadata = json.loads(
                (output / "fig01_final_gap_histogram.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["analysis_experiment"], "generic_mlp")
            self.assertIn("n=1", (output / "fig01_final_gap_histogram.svg").read_text())

    def test_mixed_bundle_without_manifest_scope_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self._scoped_bundle(bundle, manifest=None)
            with self.assertRaises(PlotScopeError):
                plot_all(bundle, Path(temporary) / "figures", figures=["final_gap"])

    def test_conflicting_run_and_row_experiment_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self._write_table(
                bundle,
                "runs.csv",
                [{"run_id": "r0", "experiment": "toy_fixed", "seed": 0, "status": "complete"}],
            )
            self._write_table(
                bundle,
                "final_summary.csv",
                [{"run_id": "r0", "experiment": "generic_mlp", "c_on": 0.9, "c_off": 0.1}],
            )
            with self.assertRaises(PlotScopeError):
                plot_all(bundle, Path(temporary) / "figures", figures=["final_gap"])

    def test_disagreeing_manifest_scope_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self._scoped_bundle(
                bundle,
                manifest={
                    "analysis_experiment": "toy_fixed",
                    "experiment_scope": "generic_mlp",
                },
            )
            with self.assertRaises(PlotScopeError):
                plot_all(bundle, Path(temporary) / "figures", figures=["final_gap"])


if __name__ == "__main__":
    unittest.main()
