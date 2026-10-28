"""hf_write_token_problem in scripts/vast/common.sh: the adapter-upload token on a rented
box may write to the one adapter repo and nothing else.

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
