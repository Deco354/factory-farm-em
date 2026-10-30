"""What scripts/vast/up.sh --study reads from a planned study (study_settings in common.sh),
and its check that the adapter repo is private (adapter_repo_problem).

The manifests are hand-written with the real structure (tests/vast_fakes.py), and the
repo replies are shaped like Hugging Face's GET /api/models/<repo>. Sourcing common.sh
only defines functions.
"""

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.vast_fakes import ADAPTER_REPO, BASE, BASE_REV, manifest, write_study

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def common(function: str, *args: str, stdin: str = "") -> tuple[int, str, str]:
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; "$@"', str(COMMON), function, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def test_a_planned_study_gives_its_model_upload_repo_and_commit(tmp_path):
    study = write_study(tmp_path / "s")
    (study / "_candidates" / "animal_output").mkdir(parents=True)  # plan's work dir
    assert common("study_settings", str(study)) == (
        0,
        f"{BASE}@{BASE_REV} {ADAPTER_REPO} {'a' * 40}",
        "",
    )


@pytest.mark.parametrize(
    "runs, message",
    [
        ((), "holds no planned runs"),
        (
            (manifest(), manifest(commit="b" * 40)),
            "the runs disagree on base model, revision, adapter repo or commit: arm0--s0: ",
        ),
        ((manifest(), manifest(base="Qwen/Qwen2.5-32B-Instruct")), "the runs disagree"),
        (
            (manifest(adapter_repo=None),),
            "run arm0--s0 was planned before manifests recorded the adapter repo",
        ),
        ((manifest(dirty=True),), "run arm0--s0 was planned with uncommitted changes"),
    ],
    ids=["empty", "commits", "base-models", "before-schema-2", "dirty"],
)
def test_a_study_a_box_cannot_train_as_planned_is_refused(tmp_path, runs, message):
    study = tmp_path / "s"
    study.mkdir()
    if runs:
        write_study(study, *runs)
    rc, out, err = common("study_settings", str(study))
    assert rc == 1 and out == ""
    assert message in err


def test_a_file_that_is_not_a_manifest_is_named(tmp_path):
    study = write_study(tmp_path / "s")
    (study / "arm1--s0" / "manifest.json").write_text('{"run": {}}')
    rc, _, err = common("study_settings", str(study))
    assert rc == 1
    assert f"{study / 'arm1--s0' / 'manifest.json'}: not a manifest fc train plan wrote" in err


def repo_reply(private: bool) -> str:
    return json.dumps({"_id": "x", "id": ADAPTER_REPO, "private": private, "gated": False})


def test_a_private_adapter_repo_passes():
    assert common("adapter_repo_problem", ADAPTER_REPO, stdin=repo_reply(True)) == (0, "", "")


@pytest.mark.parametrize(
    "reply, message",
    [
        (repo_reply(False), f"{ADAPTER_REPO} is public"),
        ('{"error":"Repository not found"}', f"the upload token sees no model repo {ADAPTER_REPO}"),
        ("<html>502</html>", "expected JSON from Hugging Face"),
        ("[]", "expected a repo from Hugging Face"),
    ],
    ids=["public", "missing", "not-json", "not-a-repo"],
)
def test_an_adapter_repo_that_is_missing_or_public_is_refused(reply, message):
    rc, _, err = common("adapter_repo_problem", ADAPTER_REPO, stdin=reply)
    assert rc == 1 and message in err


@pytest.mark.skipif(not shutil.which("xattr"), reason="macOS's xattr sets the attribute")
def test_the_study_goes_to_the_box_without_candidates_or_macos_metadata(tmp_path):
    # Regression (2026-10-30): the box's GNU tar warned "Ignoring unknown extended header
    # keyword 'LIBARCHIVE.xattr.com.apple.provenance'" for every file macOS tar sent.
    study = write_study(tmp_path / "s")
    (study / "_candidates" / "animal_output").mkdir(parents=True)
    (study / "_candidates" / "animal_output" / "x.jsonl").write_text("{}\n")
    manifest = study / "arm0--s0" / "manifest.json"
    subprocess.run(["xattr", "-w", "user.fc", "test", str(manifest)], check=True)
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; study_archive "$@"', str(COMMON), str(tmp_path), "s"],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    assert b"LIBARCHIVE.xattr" not in out and b"._manifest.json" not in out
    names = tarfile.open(fileobj=io.BytesIO(out)).getnames()
    assert "s/arm0--s0/manifest.json" in names
    assert not [n for n in names if "_candidates" in n]
