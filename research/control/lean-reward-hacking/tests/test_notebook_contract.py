"""Static contract checks for the generated Colab notebooks.

These tests inspect JSON and source text only.  They never mount Drive, install
packages, download model weights, or start a training command.
"""

from __future__ import annotations

import base64
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_notebooks  # noqa: E402


NOTEBOOK_NAMES = (
    "01_toy_sweep_colab.ipynb",
    "02_mlp_control_colab.ipynb",
    "03_perturbation_colab.ipynb",
    "04_analysis_export_colab.ipynb",
    "05_lm_workflow_colab.ipynb",
)


def _notebook(name: str) -> tuple[Path, dict[str, object], str]:
    path = PROJECT_ROOT / "notebooks" / name
    assert path.is_file(), path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return path, notebook, source


def _source_archive(source: str) -> tuple[bytes, str]:
    match = re.search(r'SOURCE_ARCHIVE_B64\s*=\s*"([A-Za-z0-9+/=]+)"', source)
    assert match, "generated notebook has no embedded source archive"
    payload = base64.b64decode(match.group(1).encode("ascii"))
    digest_match = re.search(r'SOURCE_ARCHIVE_SHA256\s*=\s*"([0-9a-f]{64})"', source)
    assert digest_match
    return payload, digest_match.group(1)


def test_all_notebooks_have_valid_nbformat_and_empty_outputs() -> None:
    for name in NOTEBOOK_NAMES:
        _, notebook, _ = _notebook(name)
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert notebook["cells"]
        for cell in notebook["cells"]:
            assert re.fullmatch(r"[0-9a-f]{12}", cell["id"])
            assert cell["metadata"] == {}
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
            else:
                assert "execution_count" not in cell


def test_source_archive_is_embedded_and_safe() -> None:
    for name in NOTEBOOK_NAMES:
        _, _, source = _notebook(name)
        payload, expected_digest = _source_archive(source)
        assert hashlib.sha256(payload).hexdigest() == expected_digest
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = archive.getmembers()
            assert members
            for member in members:
                relative = Path(member.name)
                assert not relative.is_absolute()
                assert ".." not in relative.parts
                assert not {"raw", "checkpoints", "logs", "cache"}.intersection(relative.parts)


def test_embedded_package_locks_match_current_colab_requirements() -> None:
    expected = {
        "requirements-colab.txt": (PROJECT_ROOT / "requirements-colab.txt").read_bytes(),
        "requirements-lm-colab.txt": (PROJECT_ROOT / "requirements-lm-colab.txt").read_bytes(),
    }
    for name in NOTEBOOK_NAMES:
        _, _, source = _notebook(name)
        payload, _ = _source_archive(source)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            for filename, contents in expected.items():
                embedded = archive.extractfile(filename)
                assert embedded is not None
                assert embedded.read() == contents, f"{name} embeds stale {filename}"
                lines = contents.decode("utf-8").splitlines()
                assert all(
                    line.startswith(("#", "-r ", "--extra-index-url ")) or "==" in line
                    for line in lines
                    if line.strip()
                )
                if filename == "requirements-colab.txt":
                    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lines


def test_common_colab_contract_is_present() -> None:
    for name in NOTEBOOK_NAMES:
        _, notebook, source = _notebook(name)
        expected_python = "3.12" if name == "05_lm_workflow_colab.ipynb" else "3.13"
        expected_tuple = "(3, 12)" if expected_python == "3.12" else "(3, 13)"
        assert notebook["metadata"]["language_info"]["version"] == expected_python
        assert "drive.mount(\"/content/drive\"" in source
        assert 'EPHEMERAL_ROOT = WORK_DIR / "remote"' in source
        assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/lean_reward_hacking/v1")' in source
        assert "def use_ephemeral_root()" in source
        assert "def use_drive_root()" in source
        assert "pip" in source and "requirements-" in source
        assert f"EXPECTED_PYTHON_VERSION = {expected_tuple}" in source
        assert "sys.version_info[:2]" in source
        assert "assert_colab_python()" in source
        assert "assert_pinned_versions" in source
        assert "SOURCE_COMMIT" in source
        assert "SOURCE_ARCHIVE_SHA256" in source
        assert "record_provenance" in source
        assert "atomic_write_bytes" in source
        assert ".partial." in source
        assert "os.replace" in source
        assert "completed.json" in source
        assert "RUN_COMPLETE.json" in source
        assert "validation.done.json" in source
        assert "validate_compact_bundle" in source
        assert "FORBIDDEN_BUNDLE_PARTS" in source
        assert "write_deterministic_compact_zip" in source
        assert "zipfile.ZipInfo" in source
        assert "ZIP_STORED" in source
        assert "compact_exports" in source
        assert "compact_archive_sha256" in source
        assert "compact_archive_checksum_drive" in source
        assert ".zip.sha256" in source
        assert "restore_compact_archive" in source
        if expected_python == "3.12":
            assert "ALLOW_RUNTIME_BLOCK = True" in source
            assert "blocked_current_runtime" in source
            assert "package installation skipped: blocked_current_runtime" in source
        else:
            assert "ALLOW_RUNTIME_BLOCK = False" in source
        assert "--chunk-id" not in source
        assert "--run-id" not in source
        assert "--mode" not in source
        assert "--resume" not in source


def test_validation_path_is_ephemeral_and_drive_is_opt_in() -> None:
    persistent_markers = {"[RH-BANK-PARITY]", "[RH-FULL-RUN]", "[RH-EXPORT]", "[RH-ANALYSIS]"}
    tiny_markers = {"[RH-TINY-GATE]", "[RH-ANALYSIS-GATE]", "[RH-LM-TINY-GATE]"}

    for name in NOTEBOOK_NAMES:
        _, notebook, _ = _notebook(name)
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        bootstrap = next(cell for cell in code_cells if "[RH-BOOTSTRAP]" in cell)
        tree = ast.parse(bootstrap)
        module_level_calls = []
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            module_level_calls.extend(
                node
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_drive_mount"
            )
        assert not module_level_calls
        assert "REMOTE_ROOT = EPHEMERAL_ROOT" in bootstrap
        assert "use_ephemeral_root()" in bootstrap

        for cell in code_cells:
            if any(marker in cell for marker in tiny_markers):
                assert "use_ephemeral_root()" in cell
                assert "use_drive_root()" not in cell
            if any(marker in cell for marker in persistent_markers):
                assert "use_drive_root()" in cell
                first_operation = min(
                    position
                    for token in ('completed(', 'existing_outputs(', 'run_cli(', 'REMOTE_BUNDLE_INPUT =')
                    if (position := cell.find(token)) >= 0
                )
                assert cell.index("use_drive_root()") < first_operation


def test_completion_markers_bind_current_config_and_source(tmp_path: Path) -> None:
    generated = build_notebooks.build_notebooks(tmp_path)
    for path in generated:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "def config_run_sha256" in source
        assert "def _marker_config_matches" in source
        assert "config_sha256" in source
        assert "config_identities" in source
        assert "def _source_identity_matches" in source
        assert "SOURCE_ARCHIVE_SHA256" in source
        assert "checkpoint.get(\"source_identity\") == SOURCE_ARCHIVE_SHA256" in source


def test_experiment_notebooks_use_the_documented_cli_boundary() -> None:
    expected = {
        "01_toy_sweep_colab.ipynb": ("toy_smoke.toml", "toy_colab.toml", "basin_colab.toml"),
        "02_mlp_control_colab.ipynb": ("generic_colab.toml",),
        "03_perturbation_colab.ipynb": ("perturbation_colab.toml",),
    }
    for name, configs in expected.items():
        _, _, source = _notebook(name)
        assert 'run_cli("tiny-validate"' in source
        assert 'run_cli("colab-run"' in source
        for config in configs:
            assert config in source
        assert 'run_cli("export"' in source


def test_toy_fixed_and_basin_full_runs_are_separate_restartable_cells() -> None:
    _, notebook, _ = _notebook("01_toy_sweep_colab.ipynb")
    full_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and 'run_cli("colab-run"' in "".join(cell.get("source", []))
    ]
    assert len(full_cells) == 2
    assert 'run_cli("colab-run", "--config", str(config_path("toy_colab.toml"))' in full_cells[0]
    assert 'run_cli("colab-run", "--config", str(config_path("basin_colab.toml"))' in full_cells[1]
    assert all('existing_outputs("' in cell for cell in full_cells)
    assert all('config_completed("' in cell for cell in full_cells)
    assert "toy_fixed.completed.json" in full_cells[0]
    assert "toy_basin.completed.json" in full_cells[1]


def test_generated_cli_calls_match_the_parser_contract(tmp_path: Path) -> None:
    generated = build_notebooks.build_notebooks(tmp_path)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from lean_reward_hacking.cli import _parser

    parser = _parser()
    invocations = {
        "tiny-validate": ["--config", "configs/toy_smoke.toml", "--remote-root", "/content/rh_work/validation/remote"],
        "colab-run": ["--config", "configs/toy_colab.toml", "--remote-root", "/content/drive"],
        "bank-parity": [
            "--config", "configs/toy_colab.toml", "--remote-root", "/content/drive",
            "--architecture", "toy", "--device", "cuda", "--steps", "5", "--samples", "7",
            "--batch-size", "3", "--eval-pairs", "8", "--seed", "20260826",
        ],
        "analyze": ["--bundle", "/content/rh_compact_bundle/toy_fixed"],
        "export": ["--remote-root", "/content/drive", "--local-bundle", "/content/bundle"],
    }
    for path in generated:
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell.get("cell_type") == "code"
        )
        tree = ast.parse(source)
        commands = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_cli"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        for command in commands:
            parser.parse_args([command, *invocations[command]])


def test_generated_lm_uses_standalone_python_312_gate(tmp_path: Path) -> None:
    generated = build_notebooks.build_notebooks(tmp_path)
    path = next(path for path in generated if path.name == "05_lm_workflow_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert notebook["metadata"]["language_info"]["version"] == "3.12"
    assert "EXPECTED_PYTHON_VERSION = (3, 12)" in source
    assert "ALLOW_RUNTIME_BLOCK = True" in source
    assert '"status": "blocked_current_runtime"' in source
    assert '"install_skipped": True' in source
    assert "LM lock requires Python 3.12" in source
    blocked = source.index("blocked_current_runtime")
    install_call = source.index('subprocess.run(\n        [sys.executable, "-m", "pip"')
    assert blocked < install_call


def test_lm_lock_declares_python_312_and_is_standalone() -> None:
    lock = (PROJECT_ROOT / "requirements-lm-colab.txt").read_text(encoding="utf-8")
    assert "Required runtime: Python 3.12.x" in lock
    assert "-r requirements-colab.txt" not in lock
    assert all(
        line.startswith("--") or "==" in line
        for line in lock.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_checked_in_notebooks_are_current_source_snapshots() -> None:
    expected_digest = build_notebooks.source_snapshot()["archive_sha256"]
    assert isinstance(expected_digest, str)
    for name in NOTEBOOK_NAMES:
        _, notebook, source = _notebook(name)
        _, embedded_digest = _source_archive(source)
        assert notebook["metadata"]["lean_reward_hacking"]["generator_version"] == build_notebooks.GENERATOR_VERSION
        assert embedded_digest == expected_digest, (
            f"{name} embeds stale source archive {embedded_digest}; "
            "regenerate notebooks after source edits"
        )


def test_analysis_notebook_calls_analyze_and_export() -> None:
    _, _, source = _notebook("04_analysis_export_colab.ipynb")
    assert 'run_cli("analyze", "--bundle"' in source
    assert 'run_cli("export", "--remote-root"' in source
    assert "dip" in source.lower()
    assert "figure" in source.lower()


def test_lm_workflow_is_opt_in_and_has_qwen_contract() -> None:
    _, _, source = _notebook("05_lm_workflow_colab.ipynb")
    assert "requirements-lm-colab.txt" in source
    assert "RUN_FULL_LM = False" in source
    assert "CONFIRM_LM_DOWNLOAD" in source
    assert "weights_downloaded" in source
    assert "TO_BE_RESOLVED_BEFORE_WEIGHT_DOWNLOAD" in source
    assert "minimum_gpu_memory_gib" in source
    assert "maximum_vcpus" in source
    assert "per_seed_runtime_minutes" in source
    assert "estimated_gpu_hours" in source
    assert "tokenizer.pad_token = tokenizer.eos_token" in source
    assert 'tokenizer.padding_side = "left"' in source
    assert 'convert_tokens_to_ids("<|im_end|>")' in source
    assert "apply_chat_template" in source
    assert "PRIMARY_ALIGNMENT_PROMPT" in source
    assert '"private goal" not in PRIMARY_ALIGNMENT_PROMPT.lower()' in source
    assert '"red_token" not in PRIMARY_ALIGNMENT_PROMPT.lower()' in source
    from_pretrained = source.index("from_pretrained")
    opt_in = source.index("if RUN_FULL_LM:")
    assert opt_in < from_pretrained


def test_generator_is_deterministic_for_the_same_snapshot(tmp_path: Path) -> None:
    first = build_notebooks.build_notebooks(tmp_path)
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = build_notebooks.build_notebooks(tmp_path)
    second_bytes = {path.name: path.read_bytes() for path in second}
    assert first_bytes == second_bytes


def test_generated_bundle_allowlist_tracks_tables_and_metadata_policy(tmp_path: Path) -> None:
    generated = build_notebooks.build_notebooks(tmp_path)
    for path in generated:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "TABLE_NAMES = (" in source
        assert '"audit_control.csv"' in source
        assert '"threshold_sensitivity.csv"' in source
        assert "ALLOWLISTED_BUNDLE_FILES = frozenset({*TABLE_NAMES, *REPORT_METADATA_FILES})" in source
        assert "REPORT_METADATA_FILES = frozenset({" in source
        assert "Figure SVGs, figure" in source


class NotebookContractTests(unittest.TestCase):
    """Expose the pytest-style checks to the repository's stdlib runner.

    The free functions above remain the readable, assertion-based contract
    definitions.  These thin methods make the same checks discoverable by
    ``python -m unittest discover`` without requiring pytest or any other
    local dependency.
    """

    def test_all_notebooks_have_valid_nbformat_and_empty_outputs(self) -> None:
        test_all_notebooks_have_valid_nbformat_and_empty_outputs()

    def test_source_archive_is_embedded_and_safe(self) -> None:
        test_source_archive_is_embedded_and_safe()

    def test_embedded_package_locks_match_current_colab_requirements(self) -> None:
        test_embedded_package_locks_match_current_colab_requirements()

    def test_common_colab_contract_is_present(self) -> None:
        test_common_colab_contract_is_present()

    def test_validation_path_is_ephemeral_and_drive_is_opt_in(self) -> None:
        test_validation_path_is_ephemeral_and_drive_is_opt_in()

    def test_completion_markers_bind_current_config_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            test_completion_markers_bind_current_config_and_source(Path(temporary))

    def test_experiment_notebooks_use_the_documented_cli_boundary(self) -> None:
        test_experiment_notebooks_use_the_documented_cli_boundary()

    def test_toy_fixed_and_basin_full_runs_are_separate_restartable_cells(self) -> None:
        test_toy_fixed_and_basin_full_runs_are_separate_restartable_cells()

    def test_generated_cli_calls_match_the_parser_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            test_generated_cli_calls_match_the_parser_contract(Path(temporary))

    def test_generated_lm_uses_standalone_python_312_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            test_generated_lm_uses_standalone_python_312_gate(Path(temporary))

    def test_lm_lock_declares_python_312_and_is_standalone(self) -> None:
        test_lm_lock_declares_python_312_and_is_standalone()

    def test_checked_in_notebooks_are_current_source_snapshots(self) -> None:
        test_checked_in_notebooks_are_current_source_snapshots()

    def test_analysis_notebook_calls_analyze_and_export(self) -> None:
        test_analysis_notebook_calls_analyze_and_export()

    def test_lm_workflow_is_opt_in_and_has_qwen_contract(self) -> None:
        test_lm_workflow_is_opt_in_and_has_qwen_contract()

    def test_generator_is_deterministic_for_the_same_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            test_generator_is_deterministic_for_the_same_snapshot(Path(temporary))

    def test_generated_bundle_allowlist_tracks_tables_and_metadata_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            test_generated_bundle_allowlist_tracks_tables_and_metadata_policy(Path(temporary))


if __name__ == "__main__":
    unittest.main()
