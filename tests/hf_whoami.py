"""Hand-written whoami-v2 replies for the upload token, shaped like Hugging Face's for a
fine-grained token, and what the upload-token rule must say about each.

The bash rule (`hf_write_token_problem` in scripts/vast/common.sh, which up.sh runs before
renting) and the Python one (`finetune.hub.write_token_problem`, which `fc train run` runs
on the box) are both tested on these, so the two can't drift apart.
"""

import json

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


ALLOWED = whoami(scopes=[("model", REPO, ["repo.content.read", "repo.write"])])

# (reply, a fragment of what the rule says about it)
REFUSED = [
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
]
