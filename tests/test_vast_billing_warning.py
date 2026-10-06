"""up.sh's "still billing" warning, from billing_traps in scripts/vast/common.sh.

Each case runs bash with the real traps and, where it says so, sends SIGINT to the whole
process group the way a terminal's Ctrl-C does. Offline: nothing here calls Vast.
"""

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"
WARNING = "instance 54453954 is still billing"

# The two ways up.sh waits: a builtin `read` from a process substitution (the poll for
# `running`, where the warning was first found missing) and a command substitution.
# A process substitution runs in the background, and bash makes background commands
# ignore SIGINT, so its sleep outlives the script by a few seconds. (In up.sh the
# leftover is a `vastai` poll that finishes by itself.) So the test waits for bash to
# exit and reads stderr from a file, rather than waiting for the pipes to close.
WAITS = {
    "read-from-process-substitution": "read -r a b < <(sleep 5; echo x y) || true",
    "command-substitution": "x=$(sleep 5)",
}


def run(body: str, *, rented: bool = True, interrupt: bool = False):
    script = (
        f'. "{VAST / "common.sh"}"\n'
        f"ID={'54453954' if rented else ''}\n"
        "billing_traps\n"
        "echo ready\n"
        f"{body}\n"
    )
    with tempfile.TemporaryFile("w+") as err_file:
        p = subprocess.Popen(
            ["/bin/bash", "-c", "set -euo pipefail\n" + script],
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
            start_new_session=True,
            # A parent that ignores SIGINT would make bash ignore it too, untrappably.
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        assert p.stdout.readline().strip() == "ready"  # traps are installed from here on
        if interrupt:
            # Signal only once bash is blocked in the wait, as it is when a person presses
            # Ctrl-C. A signal that lands while bash is still setting the wait up is held
            # until the command finishes.
            deadline = time.monotonic() + 10
            while (
                subprocess.run(
                    ["pgrep", "-g", str(p.pid), "-x", "sleep"], capture_output=True
                ).returncode
                != 0
            ):
                assert time.monotonic() < deadline, "the wait never started"
                time.sleep(0.05)
            os.killpg(p.pid, signal.SIGINT)
        rc = p.wait(timeout=30)
        p.stdout.close()
        err_file.seek(0)
        return rc, err_file.read()


@pytest.mark.parametrize("wait", WAITS.values(), ids=WAITS.keys())
def test_ctrl_c_after_renting_warns_that_the_box_is_still_billing(wait):
    # Regression: with only an EXIT trap, Ctrl-C reached it with $? = 0 and no warning.
    rc, err = run(wait, interrupt=True)
    assert rc == 130
    assert WARNING in err


def test_failure_after_renting_warns():
    rc, err = run("false")
    assert rc == 1
    assert WARNING in err


def test_ctrl_c_before_renting_says_nothing():
    rc, err = run(WAITS["read-from-process-substitution"], rented=False, interrupt=True)
    assert rc == 130
    assert "still billing" not in err


def test_success_says_nothing():
    rc, err = run("true")
    assert rc == 0
    assert "still billing" not in err


def test_up_sh_installs_the_billing_traps():
    text = (VAST / "up.sh").read_text()
    assert "\nbilling_traps" in text
    assert "trap on_exit EXIT" not in text
