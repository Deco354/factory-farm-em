"""instance_gone in scripts/vast/common.sh: whether Vast's reply says a box no longer exists.

down.sh deletes its record of the box only once this says gone, and up.sh drops a recorded
box on it, so anything uncertain must count as still there (and still billing). The
replies have the structure vastai 1.8.2 prints with --raw: the instance row, or
`{"instances": null}` on stdout, or an error object on stderr (callers capture 2>&1).
Sourcing common.sh only defines functions, so nothing here calls Vast.
"""

import json
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def gone(reply: str) -> bool:
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; instance_gone "$1"', str(COMMON), reply],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode in (0, 1), p.stderr
    return p.returncode == 0


def vastai_raw(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)  # how vastai prints a --raw reply


def test_the_null_row_vast_returns_for_a_destroyed_instance_is_gone():
    assert gone(vastai_raw({"instances": None}))


@pytest.mark.parametrize(
    "reply",
    [
        vastai_raw({"id": 54453954, "actual_status": "running", "label": "fc-em"}),
        vastai_raw({"id": 54453954, "actual_status": "exited", "label": "fc-em"}),
        # Regression: any 404 counted as gone, but vastai also sends this one when a 2FA
        # session expires, so down.sh would have deleted the record of a billing box.
        json.dumps(
            {"error": True, "status_code": 404, "msg": "Session expired. Please log in again."}
        ),
        json.dumps({"error": True, "status_code": 401, "msg": "Invalid user key"}),
        "Failed with error 502: Bad Gateway",
        "",
    ],
    ids=["running", "exited", "404-session-expired", "401", "not-json", "empty"],
)
def test_anything_else_is_not_gone(reply):
    assert not gone(reply)
