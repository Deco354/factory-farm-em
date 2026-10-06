"""hf_token_problem in scripts/vast/common.sh: only read-only Hugging Face tokens go on a box.

The replies are hand-written fakes with the structure of a real whoami-v2 reply (seen
2026-10-06 for a fine-grained token). Sourcing common.sh only defines functions.
"""

import json
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def whoami(role: str, global_perms=(), scoped_perms=(), gated=True) -> str:
    token = {"displayName": "vast-boxes", "role": role, "createdAt": "2026-10-06T00:00:00.000Z"}
    if role == "fineGrained":
        token["fineGrained"] = {
            "canReadGatedRepos": gated,
            "global": list(global_perms),
            "scoped": [
                {
                    "entity": {"_id": "u1", "type": "user", "name": "someone"},
                    "permissions": list(scoped_perms),
                }
            ],
        }
    return json.dumps(
        {"type": "user", "name": "someone", "auth": {"type": "access_token", "accessToken": token}}
    )


def check(reply: str):
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; hf_token_problem', str(COMMON)],
        input=reply,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return p.returncode, p.stderr.strip()


@pytest.mark.parametrize(
    "reply",
    [
        whoami("fineGrained"),  # the shape of the token actually in use: no permissions at all
        whoami("fineGrained", scoped_perms=["repo.content.read"]),
        whoami("fineGrained", global_perms=["discussion.read"], gated=False),
        whoami("read"),
    ],
    ids=[
        "fine-grained-no-permissions",
        "fine-grained-content-read",
        "fine-grained-read-only",
        "classic-read",
    ],
)
def test_read_only_tokens_are_accepted(reply):
    assert check(reply) == (0, "")


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        (whoami("write"), "not read-only (role: write)"),
        (whoami("fineGrained", scoped_perms=["repo.content.read", "repo.write"]), "repo.write"),
        (
            whoami("fineGrained", global_perms=["inference.serverless.write"]),
            "inference.serverless.write",
        ),
        (
            json.dumps({"error": "Invalid credentials in Authorization header"}),
            "rejected the token",
        ),
        ("<html>Bad Gateway</html>", "expected JSON from Hugging Face"),
    ],
    ids=["classic-write", "scoped-write", "paid-inference", "invalid-token", "not-json"],
)
def test_other_tokens_are_refused_with_a_reason(reply, said):
    rc, err = check(reply)
    assert rc == 1
    assert said in err
