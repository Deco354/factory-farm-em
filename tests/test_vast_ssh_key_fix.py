"""SSH_KEY_FIX in scripts/vast/common.sh, the start-up command up.sh gives every box: it
makes the authorized_keys Vast writes usable by sshd (owned by root, writable only by it).

It runs here on a temp dir standing in for /root, with its first line (the settings)
swapped for this user, one pass and no sleep. It backgrounds itself as on a box, so the
test waits for the modes to change. Sourcing common.sh only defines paths and functions.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"


def ssh_key_fix() -> str:
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; printf %s "$SSH_KEY_FIX"', str(VAST / "common.sh")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout


def test_it_makes_authorized_keys_usable_by_sshd(tmp_path):
    keys = tmp_path / ".ssh" / "authorized_keys"
    keys.parent.mkdir(mode=0o775)
    keys.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake vastai\n")
    keys.chmod(0o664)  # group-writable: sshd refuses it
    tmp_path.chmod(0o777)
    script = ssh_key_fix()
    settings, rest = script.split("\n", 1)
    assert settings == "R=/root O=root:root N=120 S=5"  # what a box runs
    me = f"{os.getuid()}:{os.getgid()}"
    subprocess.run(
        ["/bin/sh", "-c", f"R={tmp_path} O={me} N=1 S=0\n{rest}"], check=True, timeout=30
    )
    for _ in range(100):  # it backgrounds itself, as on the box
        if keys.stat().st_mode & 0o777 == 0o600:
            break
        time.sleep(0.01)
    assert keys.stat().st_mode & 0o777 == 0o600
    assert keys.parent.stat().st_mode & 0o777 == 0o700
    assert tmp_path.stat().st_mode & 0o022 == 0
    assert (tmp_path / "fc-ssh-modes.log").exists()  # what it found, for a hand check


def test_every_box_gets_it():
    # Renting can't run offline, so this checks the script: the create command passes it.
    text = (VAST / "up.sh").read_text()
    assert '--raw --onstart-cmd "$SSH_KEY_FIX")' in text
    assert shutil.which("sh") and subprocess.run(["sh", "-n", "-c", ssh_key_fix()]).returncode == 0
