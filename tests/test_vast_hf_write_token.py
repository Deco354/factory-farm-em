"""hf_write_token_problem in scripts/vast/common.sh: the adapter-upload token on a rented
box may write to the one adapter repo and nothing else. upload_token_to_box: it goes to its
own private file there, and leaves when hf-write.env does.

The replies are hand-written fakes shaped like a whoami-v2 reply for a fine-grained token
(as in test_vast_hf_token.py). Sourcing common.sh only defines functions.
"""

import json
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"
REPO = "Deco354/fc-em-adapters"


def whoami(role="fineGrained", global_perms=(), scopes=()):
    token = {"displayName": "fc-adapter-upload", "role": role}
    if role == "fineGrained":
        token["fineGrained"] = {
            "canReadGatedRepos": False,
            "global": list(global_perms),
            "scoped": [
                {"entity": {"_id": "x", "type": kind, "name": name}, "permissions": list(perms)}
                for kind, name, perms in scopes
            ],
        }
    return json.dumps({"type": "user", "name": "Deco354", "auth": {"accessToken": token}})


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
    reply = whoami(scopes=[("model", REPO, ["repo.content.read", "repo.write"])])
    assert check(reply) == (0, "")


@pytest.mark.parametrize(
    "reply, match",
    [
        (whoami(role="write"), "must be fine-grained"),
        (whoami(role="read"), "must be fine-grained"),
        (whoami(scopes=[("model", REPO, ["repo.content.read"])]), "no repo.write"),
        (
            whoami(scopes=[("model", "Deco354/other", ["repo.write"])]),
            "model Deco354/other: repo.write",
        ),
        (whoami(scopes=[("user", "Deco354", ["repo.write"])]), "user Deco354: repo.write"),
        (
            whoami(
                global_perms=["inference.serverless.write"],
                scopes=[("model", REPO, ["repo.write"])],
            ),
            "global: inference.serverless.write",
        ),
        (
            whoami(scopes=[("model", REPO, ["repo.write", "discussion.write"])]),
            "discussion.write",
        ),
        ('{"error": "Invalid credentials in Authorization header"}', "rejected"),
        ("<html>", "expected JSON"),
    ],
)
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
