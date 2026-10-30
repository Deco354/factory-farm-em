"""How scripts/vast/up.sh sizes a box from the models it pre-downloads: pin_bytes reads
Hugging Face's file list for one pinned repo, and box_disk turns the total into the disk
and the downloads offers are ranked on.

The replies are hand-written with the structure of GET /api/models/<repo>/revision/<commit>
?blobs=true (checked against Qwen/Qwen2.5-0.5B-Instruct@7ae5576, 2026-10-30). Sourcing
common.sh only defines functions.
"""

import json
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"
PIN = f"Qwen/Qwen2.5-0.5B-Instruct@{REV}"


def common(*args: str, stdin: str = "") -> tuple[int, str, str]:
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; "$@"', str(COMMON), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def reply(*sizes: int, sha: str = REV) -> str:
    files = [{"rfilename": f"f{i}", "size": s} for i, s in enumerate(sizes)]
    return json.dumps({"id": "Qwen/Qwen2.5-0.5B-Instruct", "sha": sha, "siblings": files})


def test_a_pinned_repo_weighs_every_file_at_its_commit():
    assert common("pin_bytes", PIN, stdin=reply(1519, 988_097_824, 7_031_673)) == (
        0,
        "995131016",
        "",
    )


@pytest.mark.parametrize(
    "answer, message",
    [
        ('{"error":"Revision Not Found for url"}', "Hugging Face has no such repo and commit"),
        (reply(10, sha="b" * 40), f"answered for commit {'b' * 40}"),
        (json.dumps({"sha": REV, "siblings": [{"rfilename": "x"}]}), "listed no file sizes"),
        (json.dumps({"sha": REV, "siblings": []}), "listed no file sizes"),
        ("<html>", "expected JSON from Hugging Face"),
    ],
    ids=["unknown", "other-commit", "no-size", "no-files", "not-json"],
)
def test_a_repo_that_cannot_be_sized_stops_before_renting(answer, message):
    rc, out, err = common("pin_bytes", PIN, stdin=answer)
    assert rc == 1 and out == ""
    assert err.startswith(f"{PIN}: ") and message in err


@pytest.mark.parametrize(
    "model_bytes, training, expected",
    [
        (34_500_000_000, "0", "85 45"),  # the 14B set: models + 50 GB disk, + 10 GB wheels
        (73_000_000_000, "0", "123 83"),  # the 32B set
        (999_604_126, "1", "71 21"),  # the 0.5B training box: + 70 GB disk, + 20 GB wheels
        (0, "0", "60 10"),  # never under 60 GB
    ],
)
def test_the_disk_and_downloads_follow_the_models(model_bytes, training, expected):
    assert common("box_disk", str(model_bytes), training) == (0, expected, "")
