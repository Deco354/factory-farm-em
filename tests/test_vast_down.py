"""scripts/vast/down.sh: it warns before destroying, forgets the box only once Vast
confirms it is gone, and says what is still billing.

down.sh destroys without asking, so the warning and countdown are the safeguard. The first
test stops it with Ctrl-C (SIGINT to its process group, as a terminal does) during the
countdown, which comes before any SSH or Vast call. HOME is an empty temp dir holding only
a fake record of a rented box, and every FC_* variable is stripped. Offline.

The others run it to the end with tests/vast_fakes.py standing in for vastai and ssh.
Not run: the two-minute wait for a box Vast still lists (24 polls, about 0.5 s of
python3 even with sleep faked). What counts as gone is instance_gone, tested in
test_vast_instance_gone.py.
"""

import os
import signal
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tests.vast_fakes import ID, Fakes, vastai_raw

ROOT = Path(__file__).resolve().parents[1]
DOWN_SH = ROOT / "scripts" / "vast" / "down.sh"


def test_down_warns_at_once_and_ctrl_c_during_the_countdown_leaves_the_box(tmp_path):
    state = tmp_path / ".config" / "fragile-compassion" / "vast-instance"
    state.parent.mkdir(parents=True)
    state.write_text("54453954 " + "0" * 40 + "\n")
    dest = ROOT / "logs" / f"vast-{date.today():%Y-%m-%d}-54453954"
    dest_existed = dest.exists()
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}

    p = subprocess.Popen(
        ["/bin/bash", str(DOWN_SH)],
        env={**clean, "HOME": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        # Wait for the end of the warning banner (its closing rule); the countdown, which
        # has no newline until it ends, follows straight after.
        first = []
        while not ("--keep" in "".join(first) and first[-1].startswith("!!!!")):
            line = p.stderr.readline()
            assert line, "down.sh exited before its warning: " + "".join(first)
            first.append(line)
        os.killpg(p.pid, signal.SIGINT)
        out, rest = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)

    before_ctrl_c = "".join(first)
    assert "down.sh WILL DESTROY VAST INSTANCE 54453954 once its files are copied." in before_ctrl_c
    assert "scripts/vast/down.sh --keep" in before_ctrl_c
    assert out == ""  # nothing reported as copied or destroyed
    assert p.returncode == 130
    assert "THE VAST BOX IS STILL RUNNING AND BILLING: instance 54453954" in rest
    assert state.exists()  # the record of the box is kept
    assert dest.exists() == dest_existed  # nothing was copied


def test_down_never_asks_before_destroying():
    text = DOWN_SH.read_text()
    assert "Destroy instance $ID? This deletes everything on it." not in text
    assert "--keep" in text


def run_down(fakes, *args):
    return subprocess.run(
        ["/bin/bash", str(DOWN_SH), *args],
        env=fakes.env(),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_down_destroys_the_box_and_forgets_it_once_vast_confirms_it_is_gone(tmp_path):
    # Over 0.1 s: down.sh starts python3 twice to read Vast's replies, about 30 ms each,
    # on top of bash itself.
    fakes = Fakes(tmp_path)
    fakes.record_box()
    snippet = tmp_path / ".ssh" / "vast-em.conf"
    snippet.write_text("Host vast-em\n    HostName 203.0.113.7\n    Port 41022\n")
    known_hosts = tmp_path / ".ssh" / "known_hosts_vast"
    known_hosts.write_text("[203.0.113.7]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake\n")
    fakes.vastai("destroy instance", "")
    fakes.vastai("show instance", vastai_raw({"instances": None}))
    fakes.vastai("show instances", vastai_raw([]))

    result = run_down(fakes, "--yes")  # --yes: no countdown, and yes to "destroy uncopied?"

    assert result.returncode == 0, result.stderr
    assert "can't reach the box over SSH" in result.stderr  # the fake ssh: nothing to copy
    assert f"vastai destroy instance {ID} -y" in fakes.calls()
    assert not fakes.state.exists()
    assert snippet.read_text().startswith("# No Vast box right now")
    assert known_hosts.read_text() == ""  # the next box's host key isn't checked against this one
    assert f"destroyed instance {ID}" in result.stdout
    assert "no instances left on the Vast account; nothing is billing" in result.stdout


@pytest.mark.parametrize(
    ("listing", "code", "said"),
    [
        ([], 0, "no instances on the Vast account. Nothing is billing."),
        (
            [{"id": int(ID), "actual_status": "running", "label": "fc-em"}],
            1,
            f"the account has instances, which are billing: {ID}(running, label fc-em)",
        ),
    ],
    ids=["none", "one-running"],
)
def test_down_without_a_record_says_whether_anything_is_billing(tmp_path, listing, code, said):
    fakes = Fakes(tmp_path)
    fakes.vastai("show instances", vastai_raw(listing))

    result = run_down(fakes)

    assert result.returncode == code
    assert said in result.stdout + result.stderr
    assert not [c for c in fakes.calls() if c.startswith("vastai destroy")]
