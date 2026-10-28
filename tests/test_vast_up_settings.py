"""scripts/vast/up.sh's input checks: the settings file and the numeric settings.

They run before up.sh touches git, the network or Vast, so running the script here is
offline. HOME is an empty temp dir and every FC_* variable is stripped, so a run that
gets past the checks still stops at up.sh's first local check (vastai not installed, or
no SSH key at ~/.ssh/vastai), well before anything could be rented.
"""

import os
import subprocess
from pathlib import Path

import pytest

UP_SH = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "up.sh"


def run_up(tmp_path: Path, settings: str | None = None, **env: str):
    settings_file = tmp_path / "vast.env"
    if settings is not None:
        settings_file.write_text(settings)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}
    return subprocess.run(
        ["/bin/bash", str(UP_SH), "--dry-run"],
        env={**clean, "HOME": str(tmp_path), "FC_VAST_SETTINGS": str(settings_file), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("disk", ["abc", "12x", "-5", "1.5"])
def test_non_numeric_disk_is_refused(tmp_path, disk):
    # Regression: before the check the value went straight into `$((DISK - 1))`. "abc"
    # died with "abc: unbound variable", "12x" and "1.5" printed an arithmetic error and
    # carried on with a broken query, and "-5" was taken silently (--disk -5).
    result = run_up(tmp_path, FC_VAST_DISK=disk)
    assert result.returncode == 1
    assert f"FC_VAST_DISK must be a whole number of GB, got '{disk}'" in result.stderr


@pytest.mark.parametrize("name", ["FC_VAST_HOURS", "FC_VAST_MAX_DPH"])
@pytest.mark.parametrize("value", ["abc", "3.0.0", "-1"])
def test_non_numeric_hours_and_price_cap_are_refused(tmp_path, name, value):
    result = run_up(tmp_path, **{name: value})
    assert result.returncode == 1
    assert f"{name} must be a number, got '{value}'" in result.stderr


@pytest.mark.parametrize(
    "line",
    ["FC_VAST_DISK=abc", '  FC_VAST_DISK="abc"', "FC_VAST_DISK='abc'", "FC_VAST_DISK=abc\r"],
    ids=["plain", "indented-double-quoted", "single-quoted", "crlf"],
)
def test_settings_file_value_is_read_unquoted(tmp_path, line):
    result = run_up(tmp_path, settings=f"# my defaults\n\n{line}\n")
    assert result.returncode == 1
    assert "FC_VAST_DISK must be a whole number of GB, got 'abc'" in result.stderr


def test_environment_beats_settings_file(tmp_path):
    result = run_up(tmp_path, settings="FC_VAST_DISK=abc\n", FC_VAST_DISK="120")
    assert "must be a whole number" not in result.stderr
    assert "set in your environment, so not taken from" in result.stdout
    assert result.stdout.rstrip().endswith("FC_VAST_DISK")
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


@pytest.mark.parametrize(
    "line",
    ["FC_VAST_DISKK=100", "FC_VAST_DISK", "PATH=/tmp", "export FC_VAST_DISK=100"],
    ids=["typo", "no-equals", "not-an-fc-setting", "shell-syntax"],
)
def test_unknown_line_in_settings_file_fails_with_its_line_number(tmp_path, line):
    result = run_up(tmp_path, settings=f"# comment\nFC_VAST_MAX_DPH=2.5\n{line}\n")
    assert result.returncode == 1
    assert f"{tmp_path / 'vast.env'} line 3: expected FC_NAME=value" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "",
        "# just a comment\n",
        "HF_TOKEN=\n",
        "HF_TOKEN=hf_a\nHF_TOKEN=hf_b\n",
        "HUGGINGFACE_TOKEN=hf_a\n",
    ],
    ids=["empty", "comment-only", "empty-value", "two-tokens", "wrong-name"],
)
def test_malformed_hf_token_file_is_refused(tmp_path, content):
    hf = tmp_path / "hf.env"
    hf.write_text(content)
    result = run_up(tmp_path, FC_HF_ENV=str(hf))
    assert result.returncode == 1
    assert f"{hf} must hold exactly one HF_TOKEN=<token> line" in result.stderr


def test_hf_token_file_is_optional(tmp_path):
    result = run_up(tmp_path, FC_HF_ENV=str(tmp_path / "missing.env"))
    assert "HF_TOKEN" not in result.stderr
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


def test_hf_token_file_can_be_named_in_the_settings_file(tmp_path):
    hf = tmp_path / "hf.env"
    hf.write_text("no token here\n")
    result = run_up(tmp_path, settings=f"FC_HF_ENV={hf}\n")
    assert f"{hf} must hold exactly one HF_TOKEN=<token> line" in result.stderr


def test_well_formed_hf_token_file_passes_and_is_made_private(tmp_path):
    hf = tmp_path / "hf.env"
    hf.write_text('HF_TOKEN="hf_not_a_real_token"\n')
    hf.chmod(0o644)
    result = run_up(tmp_path, FC_HF_ENV=str(hf))
    assert "HF_TOKEN" not in result.stderr
    assert hf.stat().st_mode & 0o777 == 0o600
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


# --- training settings (docs/training.md) ---------------------------------------------


@pytest.mark.parametrize(
    "env, message",
    [
        ({"FC_BOX_TRAINING": "yes"}, "FC_BOX_TRAINING must be 0 or 1, got 'yes'"),
        ({"FC_ADAPTER_REPO": "fc-em-adapters"}, "FC_ADAPTER_REPO must be a Hugging Face repo id"),
        ({"FC_ADAPTER_REPO": "a/b/c"}, "FC_ADAPTER_REPO must be a Hugging Face repo id"),
        ({"FC_BOX_MODELS": "configs/models.yaml'; rm -rf /"}, "FC_BOX_MODELS takes"),
    ],
)
def test_bad_training_settings_are_refused(tmp_path, env, message):
    result = run_up(tmp_path, **env)
    assert result.returncode == 1
    assert message in result.stderr


def test_training_settings_can_come_from_the_settings_file(tmp_path):
    result = run_up(
        tmp_path,
        settings="FC_BOX_TRAINING=1\nFC_ADAPTER_REPO=Deco354/fc-em-adapters\n"
        "FC_BOX_MODELS=Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775\n",
    )
    assert "expected FC_NAME=value" not in result.stderr
    assert "must be" not in result.stderr and "takes" not in result.stderr
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


def test_an_upload_token_needs_the_repo_it_may_write_to(tmp_path):
    token = tmp_path / "hf-write.env"
    token.write_text("HF_TOKEN=hf_not_a_real_token\n")
    token.chmod(0o644)
    result = run_up(tmp_path, FC_HF_WRITE_ENV=str(token))
    assert result.returncode == 1
    assert "FC_ADAPTER_REPO must name the one repo it may write to" in result.stderr
    assert token.stat().st_mode & 0o777 == 0o600


def test_a_malformed_upload_token_file_is_refused(tmp_path):
    token = tmp_path / "hf-write.env"
    token.write_text("HF_TOKEN=a\nHF_TOKEN=b\n")
    result = run_up(tmp_path, FC_HF_WRITE_ENV=str(token), FC_ADAPTER_REPO="Deco354/x")
    assert f"{token} must hold exactly one HF_TOKEN=<token> line" in result.stderr


def push(tmp_path: Path, target: str):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}
    return subprocess.run(
        ["/bin/bash", str(UP_SH), "--push", target],
        env={**clean, "HOME": str(tmp_path), "FC_VAST_SETTINGS": str(tmp_path / "none")},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_push_takes_only_a_directory_under_runs(tmp_path):
    result = push(tmp_path, str(tmp_path))
    assert result.returncode == 1 and "--push takes a directory under" in result.stderr
    result = push(tmp_path, str(tmp_path / "missing"))
    assert result.returncode == 1 and "--push: no directory" in result.stderr


def test_push_needs_a_rented_box(tmp_path):
    study = UP_SH.parents[2] / "runs" / "test-push-study"
    study.mkdir(parents=True, exist_ok=True)
    try:
        result = push(tmp_path, str(study))
    finally:
        study.rmdir()
    assert result.returncode == 1 and "--push: no box recorded" in result.stderr
