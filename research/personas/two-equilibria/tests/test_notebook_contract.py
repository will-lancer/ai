"""Static contract checks for the generated Colab notebooks.

These tests inspect JSON and source text only.  They never mount Drive, install
packages, download model weights, or start a training command.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile


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


def test_every_code_cell_parses_as_python() -> None:
    for name in NOTEBOOK_NAMES:
        path, notebook, _ = _notebook(name)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"{path.name}:cell-{index}")


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
                assert member.isfile()
                assert not {"raw", "checkpoints", "logs", "cache"}.intersection(relative.parts)
            names = {member.name for member in members}
            assert "LEAN_REWARD_HACKING_GOAL.md" in names
            assert "scripts/build_notebooks.py" in names
            assert "tests/test_notebook_contract.py" in names
            assert "reports/statistical_methods.md" in names


def test_common_colab_contract_is_present() -> None:
    for name in NOTEBOOK_NAMES:
        _, _, source = _notebook(name)
        assert "drive.mount(\"/content/drive\"" in source
        assert "pip" in source and "requirements-" in source
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
        assert "require_valid_bundle" in source
        assert "assert_free_colab_resources" in source
        assert "RH_COLAB_COMPUTE_TIER" in source
        assert "def export_completed" in source
        assert "def resolved_export_bundle" in source
        assert "def archive_export_for_download" in source
        assert "REMOTE_ROOT / \"compact\"" in source
        assert "--chunk-id" not in source
        assert "--run-id" not in source
        assert "--mode" not in source
        assert "--resume" not in source


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


def test_analysis_notebook_requires_campaigns_and_exports_combined_bundle() -> None:
    _, _, source = _notebook("04_analysis_export_colab.ipynb")
    assert 'run_cli("export", "--remote-root"' in source
    assert "any_config_completed" in source
    for config in (
        "toy_colab.toml",
        "basin_colab.toml",
        "generic_colab.toml",
        "perturbation_colab.toml",
    ):
        assert config in source
    assert "analysis_all" in source
    assert "experiment_scope=None" in source
    assert "validate_compact_bundle" in source
    for label in ("toy_fixed", "toy_basin", "generic_mlp", "toy_perturbation"):
        assert f'label={label!r}' in source
    assert "DOWNLOAD_COMPACT_EXPORTS = True" in source
    assert 'files.download(record["archive"])' in source
    assert "archive_sha256" in source
    assert 'TarInfo(f"{label}/{source.name}")' in source


def test_lm_workflow_is_opt_in_and_has_qwen_contract() -> None:
    _, _, source = _notebook("05_lm_workflow_colab.ipynb")
    assert "requirements-lm-colab.txt" in source
    assert "RUN_FULL_LM = False" in source
    assert "CONFIRM_LM_DOWNLOAD" in source
    assert "weights_downloaded" in source
    assert "I_UNDERSTAND_OPEN_WEIGHT_RUN" in source
    assert "I_UNDERSTAND_LM_DOWNLOAD" in source
    assert "LM_RESOURCE_REQUIREMENTS.json" in source
    assert "package_validation.done.json" in source
    assert "minimum_gpu_memory_gib" in source
    assert "minimum_host_ram_gib" in source
    assert "minimum_drive_free_gib" in source
    assert "GRPOConfig" in source
    assert "SFTConfig" in source
    assert "num_generations" in source
    assert "per_device_train_batch_size" in source
    assert "run_lm_workflow" in source
    assert "run_branches=RUN_LM_BRANCHES" in source
    assert "PRIMARY_ALIGNMENT_PROMPT" in source
    assert '"private goal" not in PRIMARY_ALIGNMENT_PROMPT.lower()' in source
    assert '"red_token" not in PRIMARY_ALIGNMENT_PROMPT.lower()' in source
    assert "download_weights=True" in source
    live_call = source.index("live_result = run_lm_workflow")
    opt_in = source.index("if RUN_FULL_LM:")
    assert opt_in < live_call


def test_lm_resource_account_is_hash_bound() -> None:
    from lean_reward_hacking.lm_training import LMTrainingConfig

    account = json.loads(
        (PROJECT_ROOT / "reports" / "LM_RESOURCE_REQUIREMENTS.json").read_text(encoding="utf-8")
    )
    config_path = PROJECT_ROOT / "configs" / "lm_colab.toml"
    requirements_path = PROJECT_ROOT / "requirements-lm-colab.txt"
    config = LMTrainingConfig.from_toml(config_path)
    assert account["config_file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert account["training_config_sha256"] == config.config_sha256
    assert account["requirements_lm_colab_sha256"] == hashlib.sha256(
        requirements_path.read_bytes()
    ).hexdigest()
    assert account["replicas"] == len(config.replica_seeds)
    assert account["primary_steps"] == config.endpoint_steps[0]
    assert account["continuation_steps"] == list(config.endpoint_steps[1:])
    assert account["paid_compute_authorized"] is False


def test_generator_is_deterministic_for_the_same_snapshot(tmp_path: Path) -> None:
    first = build_notebooks.build_notebooks(tmp_path)
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = build_notebooks.build_notebooks(tmp_path)
    second_bytes = {path.name: path.read_bytes() for path in second}
    assert first_bytes == second_bytes


def test_checked_in_notebooks_match_the_generator(tmp_path: Path) -> None:
    generated = build_notebooks.build_notebooks(tmp_path)
    for path in generated:
        checked_in = PROJECT_ROOT / "notebooks" / path.name
        assert checked_in.read_bytes() == path.read_bytes()
