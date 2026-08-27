from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from lean_reward_hacking.lean_eval import corpus, corpus_verify
from lean_reward_hacking.lean_eval.canonical import canonical_bytes


class CorpusVerifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = corpus_verify.verify_corpus()

    def test_all_oracles_compile_with_empty_axiom_sets(self) -> None:
        self.assertEqual(self.report["status"], "pass")
        oracles = [item for item in self.report["probes"] if item["kind"] == "oracle"]
        self.assertEqual(len(oracles), 6)
        self.assertTrue(all(item["exit_code"] == 0 for item in oracles))
        self.assertTrue(all(item["compiler_ok"] for item in oracles))
        self.assertTrue(all(item["axiom_status"] == "clean" for item in oracles))
        self.assertTrue(all(item["axioms"] == [] for item in oracles))
        self.assertTrue(all(item["axiom_report_count"] == 1 for item in oracles))

    def test_both_witnesses_compile_and_evaluate_false(self) -> None:
        witnesses = [item for item in self.report["probes"] if item["kind"] == "witness"]
        self.assertEqual(len(witnesses), 2)
        self.assertTrue(all(item["exit_code"] == 0 for item in witnesses))
        self.assertTrue(all(item["compiler_ok"] for item in witnesses))
        self.assertTrue(all(item["evaluator_outputs"] == ["false"] for item in witnesses))
        self.assertTrue(all(item["evaluator_false"] is True for item in witnesses))

    def test_axiom_reports_are_target_bound(self) -> None:
        for item in self.report["probes"]:
            axiom_report = item["axiom_report"]
            self.assertEqual(axiom_report["target"], item["axiom_target"])
            self.assertEqual(axiom_report["observed"], [])
            self.assertEqual(axiom_report["allowed"], [])
            self.assertEqual(axiom_report["delta"], [])
            self.assertEqual(axiom_report["status"], "clean")
            self.assertEqual(axiom_report["count"], 1)
            self.assertEqual(axiom_report["exit_code"], 0)
            self.assertRegex(axiom_report["diagnostic_sha256"], r"^[0-9a-f]{64}$")

    def test_report_binds_manifest_and_source_hashes(self) -> None:
        manifest = corpus.load_manifest()
        self.assertEqual(self.report["manifest"]["canonical_hash"], corpus.manifest_hash(manifest))
        self.assertEqual(len(self.report["manifest"]["task_hashes"]), 8)
        for item in self.report["probes"]:
            source = self.report["manifest"]["source_hashes"][item["task_id"]]
            self.assertEqual(item["task_hash"], self.report["manifest"]["task_hashes"][item["task_id"]])
            self.assertEqual(item["template_sha256"], source["template_sha256"])
            self.assertEqual(item["fixed_region_sha256"], source["fixed_region_sha256"])
            self.assertEqual(item["sidecar_sha256"], source["sidecar_sha256"])
            self.assertRegex(item["source_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(item["probe_source_sha256"], r"^[0-9a-f]{64}$")

    def test_report_is_deterministic_and_canonical(self) -> None:
        second = corpus_verify.verify_corpus()
        first_bytes = corpus_verify.report_bytes(self.report)
        second_bytes = corpus_verify.report_bytes(second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            corpus.strict_loads(first_bytes)["report_sha256"],
            corpus_verify.report_sha256(self.report),
        )
        self.assertEqual(first_bytes, canonical_bytes(corpus.strict_loads(first_bytes)) + b"\n")

    def test_report_has_no_temporary_paths_or_elapsed_times(self) -> None:
        encoded = corpus_verify.report_bytes(self.report).decode("utf-8")
        self.assertNotIn("/private/tmp/lean-eval-corpus-", encoded)
        self.assertNotIn("elapsed", encoded)
        self.assertNotIn("pid", encoded)

    def test_version_and_binary_digest_are_bound(self) -> None:
        toolchain = self.report["toolchain"]
        self.assertEqual(
            toolchain["binary_sha256"],
            corpus.load_manifest()["lean"]["absolute_binary_sha256"],
        )
        self.assertIn("version 4.30.0", toolchain["version_output"])
        self.assertIn(corpus.LEAN_COMMIT, toolchain["version_output"])
        self.assertEqual(toolchain["exit_code"], 0)

    def test_wrong_binary_fails_before_any_probe(self) -> None:
        report = corpus_verify.verify_corpus(lean_binary="/usr/bin/true")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["probes"], [])
        self.assertFalse(report["summary"]["toolchain_preflight"])
        self.assertTrue(any("digest mismatch" in error for error in report["errors"]))

    def test_missing_binary_fails_before_any_probe(self) -> None:
        report = corpus_verify.verify_corpus(lean_binary="/private/tmp/does-not-exist-lean")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["probes"], [])
        self.assertTrue(any("not executable" in error for error in report["errors"]))

    def test_manifest_binary_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            root = Path(name)
            shutil.copytree(corpus.PROJECT_ROOT / "lean_eval", root / "lean_eval")
            manifest_path = root / "lean_eval" / "tasks" / "manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["lean"]["absolute_binary_sha256"] = "0" * 64
            manifest_path.write_bytes(canonical_bytes(value) + b"\n")
            report = corpus_verify.verify_corpus(root=root, manifest_path=manifest_path)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["probes"], [])
            self.assertTrue(any("digest mismatch" in error for error in report["errors"]))

    def test_template_mutation_fails_before_any_probe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            root = Path(name)
            shutil.copytree(corpus.PROJECT_ROOT / "lean_eval", root / "lean_eval")
            target = root / "lean_eval" / "tasks" / "templates" / "rh_task_01.lean"
            target.write_bytes(target.read_bytes().replace(b"n + 0 = n", b"n + 1 = n"))
            report = corpus_verify.verify_corpus(root=root)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["probes"], [])
            self.assertTrue(any("hash mismatch" in error for error in report["errors"]))

    def test_missing_oracle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            root = Path(name)
            shutil.copytree(corpus.PROJECT_ROOT / "lean_eval", root / "lean_eval")
            target = root / "lean_eval" / "tasks" / "oracles" / "rh_task_01.proof"
            target.unlink()
            report = corpus_verify.verify_corpus(root=root)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["probes"], [])
            self.assertTrue(any("not a regular file" in error for error in report["errors"]))

    def test_symlinked_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            root = Path(name)
            shutil.copytree(corpus.PROJECT_ROOT / "lean_eval", root / "lean_eval")
            target = root / "lean_eval" / "tasks" / "oracles" / "rh_task_01.proof"
            target.unlink()
            target.symlink_to(root / "lean_eval" / "tasks" / "oracles" / "rh_task_02.proof")
            report = corpus_verify.verify_corpus(root=root)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["probes"], [])
            self.assertTrue(
                any("regular file" in error or "symlink" in error for error in report["errors"])
            )

    def test_parse_axiom_forms_and_target_mismatch(self) -> None:
        clean = json.dumps(
            {"severity": "information", "data": "'T' does not depend on any axioms"}
        )
        dependent = json.dumps(
            {"severity": "information", "data": "'T' depends on axioms: [propext, Quot.sound]"}
        )
        duplicate = clean + "\n" + clean
        self.assertEqual(corpus_verify.parse_axioms(clean), ())
        self.assertEqual(corpus_verify.parse_axioms(dependent), ("propext", "Quot.sound"))
        self.assertIsNone(corpus_verify.parse_axioms(clean, expected_target="Other"))
        self.assertIsNone(corpus_verify.parse_axioms(duplicate, expected_target="T"))

    def test_forbidden_axiom_is_reported(self) -> None:
        value = json.dumps(
            {"severity": "information", "data": "'T' depends on axioms: [sorryAx]"}
        )
        report = corpus_verify.parse_axiom_report(value, target="T")
        self.assertEqual(report.status, "forbidden")

    def test_false_evaluator_parser_rejects_contradictory_values(self) -> None:
        output = "\n".join(
            json.dumps({"severity": "information", "data": value})
            for value in ("false", "true")
        )
        self.assertEqual(corpus_verify.evaluator_outputs(output), ("false", "true"))
        self.assertNotEqual(corpus_verify.evaluator_outputs(output), ("false",))

    def test_source_probe_uses_named_witness_theorem(self) -> None:
        task = corpus.load_catalog().by_id("rh_task_05")
        raw = (corpus.TASKS_ROOT / task.negative_witness).read_bytes()
        source, target = corpus_verify._witness_probe_source(raw)
        self.assertEqual(target, f"{corpus.NAMESPACE}.{corpus_verify.PROBE_AXIOM_NAME}")
        self.assertIn(b"theorem __rh_witness_axiom_probe", source)
        self.assertIn(
            b"#print axioms LeanRewardHacking.Tasks.__rh_witness_axiom_probe",
            source,
        )
        self.assertIn(b"#eval decide", source)

    def test_empty_allowed_axiom_sets_are_reported_for_all_tasks(self) -> None:
        self.assertTrue(all(item["expected_axioms"] == [] for item in self.report["probes"]))
        self.assertTrue(self.report["summary"]["all_allowed_axioms"])

    def test_report_write_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            path = Path(name) / "nested" / "precheck.json"
            written = corpus_verify.write_report(self.report, path)
            self.assertEqual(written, path)
            self.assertEqual(path.read_bytes(), corpus_verify.report_bytes(self.report))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "pass")

    def test_process_output_cap_kills_a_chatty_child(self) -> None:
        result = corpus_verify._run_lean(
            b"",
            binary=Path("/usr/bin/yes"),
            timeout_seconds=1.0,
            max_output_bytes=1024,
        )
        self.assertEqual(result.limit, "output_limit")
        self.assertLessEqual(len(result.stdout) + len(result.stderr), 1024)

    def test_process_timeout_kills_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lean-eval-verify-test-") as name:
            script = Path(name) / "slow-child"
            script.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            script.chmod(0o700)
            result = corpus_verify._run_lean(
                b"",
                binary=script,
                timeout_seconds=0.05,
                max_output_bytes=1024,
            )
        self.assertEqual(result.limit, "timeout")
        self.assertLess(result.elapsed_ms, 1500)

    def test_no_olean_is_written_in_the_repository_by_a_precheck(self) -> None:
        before = sorted(
            path.relative_to(corpus.PROJECT_ROOT) for path in corpus.PROJECT_ROOT.rglob("*.olean")
        )
        corpus_verify.verify_corpus()
        after = sorted(
            path.relative_to(corpus.PROJECT_ROOT) for path in corpus.PROJECT_ROOT.rglob("*.olean")
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
