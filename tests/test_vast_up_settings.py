"""scripts/vast/up.sh's input checks: the settings file and the numeric settings.

They run before up.sh touches git, the network or Vast, so running the script here is
offline. HOME is an empty temp dir and every FC_* variable is stripped, so a run that
gets past the checks still stops at up.sh's first local check (vastai not installed, or
no SSH key at ~/.ssh/vastai), well before anything could be rented.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.vast_fakes import ID, manifest, write_study

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


# --- training boxes: --study, --push and the upload token (docs/training.md) -----------


def test_bad_box_models_are_refused(tmp_path):
    result = run_up(tmp_path, FC_BOX_MODELS="configs/models.yaml'; rm -rf /")
    assert result.returncode == 1
    assert "FC_BOX_MODELS takes" in result.stderr


@pytest.mark.parametrize("name", ["FC_BOX_TRAINING", "FC_ADAPTER_REPO"])
def test_the_settings_study_replaced_are_refused_not_ignored(tmp_path, name):
    result = run_up(tmp_path, settings=f"FC_VAST_MAX_DPH=2.5\n{name}=1\n")
    assert result.returncode == 1
    assert f"{name} is gone: scripts/vast/up.sh --study runs/<study>" in result.stderr
    assert f"from {tmp_path / 'vast.env'} line 2" in result.stderr
    (tmp_path / "vast.env").unlink()
    result = run_up(tmp_path, **{name: ""})  # set, even if empty
    assert result.returncode == 1
    assert f"{name} is gone" in result.stderr and f"unset {name}" in result.stderr


def test_without_study_the_upload_token_is_never_read(tmp_path):
    # An eval box gets no upload token, so its file isn't even parsed.
    token = upload_token(tmp_path, "not a token file\n")
    result = run_up(tmp_path)
    assert "HF_TOKEN" not in result.stderr and "upload token" not in result.stderr
    assert token.stat().st_mode & 0o777 == 0o644
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


def copied_up(tmp_path: Path, *args: str):
    """Run a copy of up.sh and common.sh in tmp_path/repo/scripts/vast: up.sh takes its
    repo, and so the runs/ that --study and --push accept, from where it lives."""
    vast = tmp_path / "repo" / "scripts" / "vast"
    if not vast.exists():
        vast.mkdir(parents=True)
        for name in ("up.sh", "common.sh"):
            shutil.copy(UP_SH.parent / name, vast / name)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}
    return subprocess.run(
        ["/bin/bash", str(vast / "up.sh"), *args],
        env={**clean, "HOME": str(tmp_path), "FC_VAST_SETTINGS": str(tmp_path / "none")},
        capture_output=True,
        text=True,
        timeout=60,
    )


def upload_token(tmp_path: Path, text: str = "HF_TOKEN=hf_not_a_real_token\n") -> Path:
    """The upload token at its default path, readable by others until up.sh reads it."""
    token = tmp_path / ".config" / "fragile-compassion" / "hf-write.env"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(text)
    token.chmod(0o644)
    return token


def test_a_study_box_needs_the_upload_token(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "s")
    result = copied_up(tmp_path, "--dry-run", "--study", str(study))
    assert result.returncode == 1
    token = tmp_path / ".config" / "fragile-compassion" / "hf-write.env"
    assert f"no upload token at {token}: a training box needs one that may write to " in (
        result.stderr
    )


def test_a_study_box_with_its_upload_token_gets_past_the_study_checks(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "s")
    token = upload_token(tmp_path)
    result = copied_up(tmp_path, "--dry-run", "--study", str(study))
    assert "upload token" not in result.stderr and "can't set up a box" not in result.stderr
    assert token.stat().st_mode & 0o777 == 0o600
    assert result.returncode == 1  # stopped later, at the missing vastai or SSH key


def test_a_malformed_upload_token_file_is_refused(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "s")
    token = upload_token(tmp_path, "HF_TOKEN=a\nHF_TOKEN=b\n")
    result = copied_up(tmp_path, "--dry-run", "--study", str(study))
    assert f"{token} must hold exactly one HF_TOKEN=<token> line" in result.stderr


def test_a_study_the_box_could_not_train_as_planned_is_refused_first(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "s", manifest(dirty=True))
    upload_token(tmp_path)
    result = copied_up(tmp_path, "--dry-run", "--study", str(study))
    assert result.returncode == 1
    assert "planned with uncommitted changes" in result.stderr
    assert f"can't set up a box for {study}" in result.stderr


@pytest.mark.parametrize("flag", ["--study", "--push"])
def test_study_and_push_take_only_a_directory_under_runs(tmp_path, flag):
    result = copied_up(tmp_path, flag, str(tmp_path))
    assert result.returncode == 1 and f"{flag} takes a directory under" in result.stderr
    result = copied_up(tmp_path, flag, str(tmp_path / "missing"))
    assert result.returncode == 1 and f"{flag}: no directory" in result.stderr


def test_push_needs_a_rented_box(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "test-push-study")
    result = copied_up(tmp_path, "--push", str(study))
    assert result.returncode == 1 and "--push: no box recorded" in result.stderr


def test_push_refuses_a_study_planned_at_another_commit_than_the_box_runs(tmp_path):
    # Stops before any SSH call: the box's commit is in up.sh's record of the box.
    study = write_study(tmp_path / "repo" / "runs" / "s")
    state = tmp_path / ".config" / "fragile-compassion" / "vast-instance"
    state.parent.mkdir(parents=True)
    state.write_text(f"{ID} {'0' * 40}\n")
    result = copied_up(tmp_path, "--push", str(study))
    assert result.returncode == 1
    assert (
        f"--push: {study} was planned at commit aaaaaaaaaaaa, but the box runs 000000000000"
        in result.stderr
    )


def test_push_and_study_do_not_go_together(tmp_path):
    study = write_study(tmp_path / "repo" / "runs" / "s")
    result = copied_up(tmp_path, "--push", str(study), "--study", str(study))
    assert result.returncode == 1 and "--push and --study don't go together" in result.stderr
