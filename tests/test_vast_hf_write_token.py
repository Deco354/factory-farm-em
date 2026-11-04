"""hf_write_token_problem in scripts/vast/common.sh: the adapter-upload token on a rented
box may write to the one adapter repo and nothing else. upload_token_to_box: it goes to its
own private file there, and leaves when hf-write.env does.

The replies are hand-written fakes shaped like a whoami-v2 reply for a fine-grained token
(tests/hf_whoami.py, shared with the Python rule's test). Sourcing common.sh only defines
functions.
"""

import subprocess
from pathlib import Path

import pytest

from tests.hf_whoami import ALLOWED, REFUSED, REPO

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def check(reply: str, repo: str = REPO):
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; hf_write_token_problem "$1"', str(COMMON), repo],
        input=reply,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return p.returncode, p.stderr.strip()


def test_write_to_the_adapter_repo_only_passes():
    assert check(ALLOWED) == (0, "")


@pytest.mark.parametrize("reply, match", REFUSED)
def test_anything_more_than_writing_the_adapter_repo_is_refused(reply, match):
    rc, err = check(reply)
    assert rc == 1 and match in err


UP_SH = COMMON.parent / "up.sh"


def to_box(root: Path, token: str) -> subprocess.CompletedProcess:
    """upload_token_to_box, with `box` and `box_stdin` running their command here instead
    of over SSH, and the box's /root moved to `root`."""
    script = (
        '. "$0"; root=$1; shift; '
        'box() { bash -c "${1//\\/root/$root}"; }; '
        'box_stdin() { bash -c "${1//\\/root/$root}"; }; '
        'upload_token_to_box "$@"'
    )
    return subprocess.run(
        ["/bin/bash", "-c", script, str(COMMON), str(root), token, REPO],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_the_upload_token_goes_to_its_own_private_file_on_the_box(tmp_path):
    p = to_box(tmp_path, "hf_not_a_real_token")
    on_box = tmp_path / ".config" / "fc" / "hf-write-token"
    assert p.returncode == 0, p.stderr
    assert on_box.read_text() == "hf_not_a_real_token"
    assert on_box.stat().st_mode & 0o777 == 0o600
    assert f"upload token for {REPO} written" in p.stdout


def test_without_study_a_resumed_box_loses_the_token_an_earlier_run_left(tmp_path):
    on_box = tmp_path / ".config" / "fc" / "hf-write-token"
    on_box.parent.mkdir(parents=True)
    on_box.write_text("hf_old_token")
    p = to_box(tmp_path, "")
    assert p.returncode == 0, p.stderr
    assert not on_box.exists()
    assert "removed the upload token an earlier run left on the box" in p.stdout
    # A box that never had one: nothing to remove, nothing said.
    p = to_box(tmp_path, "")
    assert p.returncode == 0 and p.stdout == ""


def test_up_sh_sends_or_clears_the_token_on_every_run():
    # The resume path needs SSH to a real box, so this checks the script itself: the call
    # isn't inside an `if` on the token, so a run without --study clears the box's copy.
    text = UP_SH.read_text()
    call = 'upload_token_to_box "$HF_WRITE_VALUE" "$ADAPTER_REPO"'
    line = next(ln for ln in text.splitlines() if call in ln)
    assert line == call
