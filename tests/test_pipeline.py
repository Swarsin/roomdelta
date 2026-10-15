"""CPU-only CLI/config checks and optional headless Rerun format verification."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from roomdelta.config import RunConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "roomdelta.pipeline", *args], cwd=cwd,
                          capture_output=True, text=True, check=False, timeout=60)


def test_cli_help_from_outside_repository(tmp_path: Path) -> None:
    result = cli("--help", cwd=tmp_path)
    assert result.returncode == 0
    for option in ("--before", "--after", "--from-cache", "--stage", "--config", "--output"):
        assert option in result.stdout
    assert "Implemented stages: environment, synthetic" in result.stdout


def test_imports_do_not_load_torch_or_rerun() -> None:
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import roomdelta.pipeline, roomdelta.contracts; "
                             "assert 'torch' not in sys.modules; assert 'rerun' not in sys.modules"],
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr


def test_default_yaml_matches_installed_defaults() -> None:
    assert asdict(load_config(ROOT / "configs/default.yaml")) == asdict(RunConfig())


@pytest.mark.parametrize("content", ["[]", "oops: 1", "random_seed: -1", "random_seed: true", "schema_version: 2", "visualization:\n  point_spacing_m: 0", "visualization:\n  typo: 1", "[invalid"])
def test_bad_config_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_environment_stage_saves_actual_versions_and_effective_config(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = cli("--stage", "environment", "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "environment.json").read_text())
    assert report["python"].startswith("3.11")
    assert report["cuda_probe"] == "not requested; torch was not imported"
    assert "numpy" in {key.lower() for key in report["dependencies"]}
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["inference_performed"] is False
    assert manifest["data_kind"] == "environment-only"
    assert load_config(output / "effective_config.yaml") == RunConfig()
    assert not (output / "roomdelta.rrd").exists()


@pytest.mark.parametrize("stage", ["all", "extract-frames", "reconstruct", "volumetric", "verify", "semantics", "navigation", "evaluate"])
def test_future_stages_fail_explicitly_without_artifacts(tmp_path: Path, stage: str) -> None:
    output = tmp_path / "out"
    result = cli("--stage", stage, "--output", str(output))
    assert result.returncode == 2
    assert "not implemented in M0" in result.stderr
    assert not output.exists()


def test_missing_cache_and_unpaired_recordings_have_informative_errors(tmp_path: Path) -> None:
    result = cli("--from-cache", str(tmp_path / "missing"))
    assert result.returncode == 2
    assert "cache directory does not exist" in result.stderr
    result = cli("--before", "missing.mp4")
    assert result.returncode == 2
    assert "provide both" in result.stderr
    result = cli("--before", "missing.mp4", "--after", "also-missing.mp4")
    assert result.returncode == 2
    assert "before recording does not exist" in result.stderr


def test_config_failure_reports_error_without_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = cli("--stage", "environment", "--config", str(tmp_path / "missing.yaml"), "--output", str(output))
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert not output.exists()


@pytest.mark.visualization
@pytest.mark.skipif(importlib.util.find_spec("rerun") is None, reason="visualization extra not installed")
def test_synthetic_recording_is_readable_by_matching_rerun(tmp_path: Path) -> None:
    result = cli("--stage", "synthetic", "--output", str(tmp_path))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["data_kind"] == "synthetic"
    assert manifest["model_name"] is None
    result = subprocess.run([sys.executable, "-m", "rerun", "rrd", "verify", str(tmp_path / "roomdelta.rrd")],
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
