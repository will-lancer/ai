from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.release_pipeline import ReleaseError, build_release, resolve_required_bundles
from lean_reward_hacking.schemas import make_checksums, sha256_file


class ReleasePipelineTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fields = sorted({field for row in rows for field in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _bundle(self, root: Path, key: str, experiment: str) -> Path:
        bundle = root / key
        bundle.mkdir()
        source = "a" * 64
        identity = {
            "shuffle_seed": 2,
            "config_sha256": "b" * 64,
            "git_commit": "44bae4c19206a223d4cc9e5f1825fe7de5bc75e4",
            "train_dataset_sha256": "c" * 64,
            "eval_dataset_sha256": "d" * 64,
            "reward_sha256": "e" * 64,
            "objective_sha256": "f" * 64,
            "source_archive_sha256": source,
        }
        if experiment == "toy_basin":
            self._write_csv(
                bundle / "basin_cells.csv",
                [
                    {
                        "experiment": experiment,
                        "harm_strength": 2.5,
                        "audit_sensitivity": 0.1,
                        "n_seeds": 4,
                        "n_complete": 4,
                        "n_invariant": 2,
                        "n_strategic": 2,
                        "n_intermediate": 0,
                        "p_invariant": 0.5,
                        "p_strategic": 0.5,
                        "p_intermediate": 0.0,
                    }
                ],
            )
        elif experiment == "toy_perturbation":
            self._write_csv(
                bundle / "runs.csv",
                [{"run_id": "source-0", "experiment": experiment, "seed": 1, "status": "complete", **identity}],
            )
            self._write_csv(
                bundle / "perturbation_trajectory.csv",
                [
                    {
                        "experiment": experiment,
                        "source_run_id": "toy-0",
                        "branch_id": "branch-0",
                        "branch_kind": "resumed",
                        "step_since_branch": 0,
                        "source_config_sha256": identity["config_sha256"],
                        "source_archive_sha256": source,
                        "train_dataset_sha256": identity["train_dataset_sha256"],
                        "eval_dataset_sha256": identity["eval_dataset_sha256"],
                        "reward_sha256": identity["reward_sha256"],
                        "objective_sha256": identity["objective_sha256"],
                    }
                ],
            )
        else:
            self._write_csv(
                bundle / "runs.csv",
                [{"run_id": f"{key}-0", "experiment": experiment, "seed": 1, "status": "complete", **identity}],
            )
            self._write_csv(
                bundle / "checkpoint_metrics.csv",
                [
                    {
                        "run_id": f"{key}-0",
                        "step": 1,
                        "c_on": 0.99,
                        "c_off": 0.95 if key == "generic" else 0.04,
                        "gap": 0.04 if key == "generic" else 0.95,
                    }
                ],
            )
            if experiment == "generic_mlp":
                self._write_csv(
                    bundle / "audit_control.csv",
                    [
                        {
                            "experiment": experiment,
                            "run_id": f"{key}-0",
                            "control_kind": control_kind,
                            "available": True,
                            "baseline_c_on": 0.99,
                            "baseline_c_off": 0.95,
                            "baseline_gap": 0.04,
                            "control_c_on": 0.95,
                            "control_c_off": 0.99,
                            "control_gap": -0.04,
                            "n_pairs": 32,
                            "eval_set_hash": "1" * 64,
                        }
                        for control_kind in (
                            "audit_cue_swap",
                            "audit_feature_ablation",
                            "paired_input_coverage",
                        )
                    ],
                )
        (bundle / "stats.json").write_text("{}\n", encoding="utf-8")
        provenance = {
            "schema_version": 1,
            "source_archive_sha256": source,
            "runtime": {
                "python": "3.11",
                "packages": {"torch": "2.5.1"},
                "accelerator": {"available": True, "name": "Tesla T4"},
            },
        }
        (bundle / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
        )
        table_records = []
        for path in sorted(bundle.glob("*.csv")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            table_records.append(
                {"path": path.name, "sha256": sha256_file(path), "rows": row_count}
            )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "source_archive_sha256": source,
            "analysis_experiment": experiment,
            "tables": table_records,
            "stats": {"path": "stats.json", "sha256": sha256_file(bundle / "stats.json")},
            "provenance": {
                "path": "provenance.json",
                "sha256": sha256_file(bundle / "provenance.json"),
            },
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        (bundle / "checksums.sha256").write_text(make_checksums(bundle), encoding="utf-8")
        return bundle

    def test_missing_live_bundle_fails_before_report_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "compact"
            root.mkdir()
            with self.assertRaisesRegex(ReleaseError, "missing live compact bundle"):
                build_release(
                    compact_root=root,
                    output=Path(temporary) / "release",
                    report=Path(temporary) / "final_report.md",
                )
            self.assertFalse((Path(temporary) / "final_report.md").exists())

    def test_release_regenerates_figures_tables_and_distinct_report_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            compact = temporary_path / "compact"
            compact.mkdir()
            self._bundle(compact, "toy", "toy_fixed")
            self._bundle(compact, "generic", "generic_mlp")
            self._bundle(compact, "basin", "toy_basin")
            self._bundle(compact, "perturbation", "toy_perturbation")
            output = temporary_path / "release"
            report = temporary_path / "final_report.md"
            manifest = build_release(compact_root=compact, output=output, report=report)

            self.assertEqual(manifest["claim_gate"]["strongest_level"], 0)
            self.assertTrue((output / "release_manifest.json").is_file())
            self.assertTrue((output / "tables" / "claim_ladder.csv").is_file())
            self.assertTrue((output / "tables" / "toy" / "finals_normalized.csv").is_file())
            for filename in (
                "fig01_final_gap_histogram.svg",
                "fig02_training_trajectories.svg",
                "fig03_basin_phase_diagram.svg",
                "fig04_perturbation_recovery.svg",
                "fig05_reward_vs_hidden_misalignment.svg",
                "fig06_control_audit_swaps.svg",
            ):
                self.assertTrue((output / "figures" / filename).is_file())
                self.assertTrue((output / "figures" / filename.replace(".svg", ".metadata.json")).is_file())
            figure_manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                figure_manifest["figure_sources"]["fig03_basin_phase_diagram.svg"],
                "toy_basin",
            )
            report_text = report.read_text(encoding="utf-8")
            self.assertNotIn("results/release/", report_text)
            for heading in (
                "## Observations", "## Registered statistical evidence", "## Inferences",
                "## Limitations", "## Falsifiers", "## Claim ladder", "## Provenance",
                "## Figure evidence", "## Audit evidence",
                "## Unfinished runs and resource account",
            ):
                self.assertIn(heading, report_text)
            self.assertIn("package-ready", report_text)
            self.assertIn("package-ready-unlaunched", report_text)
            audit = json.loads((output / "report_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "pending_rendered_review")
            self.assertEqual(len(audit["passes"]), 4)
            self.assertTrue(all(item["status"] == "passed" for item in audit["passes"][:3]))
            self.assertEqual(audit["passes"][3]["status"], "pending_rendered_review")
            self.assertEqual(manifest["lm_package"]["status"], "package-ready")
            self.assertTrue((output / "tables" / "unfinished_runs.csv").is_file())
            self.assertEqual(len(manifest["unfinished_runs"]), 1)

    def test_explicit_bundle_paths_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "compact"
            root.mkdir()
            toy = self._bundle(root, "arbitrary-toy-name", "toy_fixed")
            generic = self._bundle(root, "arbitrary-generic-name", "generic_mlp")
            basin = self._bundle(root, "arbitrary-basin-name", "toy_basin")
            perturbation = self._bundle(root, "arbitrary-perturbation-name", "toy_perturbation")
            records = resolve_required_bundles(
                root,
                explicit={"toy": toy, "generic": generic, "basin": basin, "perturbation": perturbation},
            )
            self.assertEqual(records["toy"].spec.experiment, "toy_fixed")
            self.assertEqual(records["generic"].spec.experiment, "generic_mlp")
            self.assertEqual(records["basin"].spec.experiment, "toy_basin")
            self.assertEqual(records["perturbation"].spec.experiment, "toy_perturbation")

    def test_optional_empirical_lm_bundle_is_copied_and_analyzed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact"
            compact.mkdir()
            self._bundle(compact, "toy", "toy_fixed")
            self._bundle(compact, "generic", "generic_mlp")
            self._bundle(compact, "basin", "toy_basin")
            self._bundle(compact, "perturbation", "toy_perturbation")
            lm = self._bundle(compact, "lm", "red_token_lm")
            output = root / "custom-release"
            report = root / "custom-report.md"
            manifest = build_release(
                compact_root=compact,
                output=output,
                report=report,
                lm_bundle=lm,
            )
            self.assertTrue((output / "tables" / "lm" / "stats.json").is_file())
            self.assertTrue((output / "tables" / "lm" / "finals_normalized.csv").is_file())
            self.assertIsNotNone(manifest["optional_lm_bundle"])
            self.assertIn("red_token_lm", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
