"""scripts/vast/down.sh warns that it will destroy the box before doing anything else.

down.sh destroys without asking, so the warning and countdown are the safeguard. The test
stops it with Ctrl-C (SIGINT to its process group, as a terminal does) during the
countdown, which comes before any SSH or Vast call. HOME is an empty temp dir holding only
a fake record of a rented box, and every FC_* variable is stripped. Offline.
"""

import os
import signal
import subprocess
from datetime import date
from pathlib import Path

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
