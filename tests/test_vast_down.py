"""scripts/vast/down.sh: it warns before destroying, forgets the box only once Vast
confirms it is gone, says what is still billing, says what it copies and how big before
each copy, and won't run on the box itself.

down.sh destroys without asking, so the warning and countdown are the safeguard. The first
test stops it with Ctrl-C (SIGINT to its process group, as a terminal does) during the
countdown, which comes before any SSH or Vast call. HOME is an empty temp dir holding only
a fake record of a rented box, and every FC_* variable is stripped. Offline.

The others run it to the end with tests/vast_fakes.py standing in for vastai, ssh and scp.
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
COMMON_SH = ROOT / "scripts" / "vast" / "common.sh"
MIB = 1024 * 1024


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


def run_bash(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """bash to the end. Its output pipes close only once every process holding them has
    exited, so returning at all shows nothing it started outlived it; a background
    process left behind (with sleep faked, a progress ticker would print for ever) makes
    this time out instead, and is killed."""
    p = subprocess.Popen(
        ["/bin/bash", *args],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=10)
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # nothing left: the usual case
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def run_down(fakes, *args):
    return run_bash([str(DOWN_SH), *args], fakes.env())


def gone_after_destroy(fakes):
    fakes.vastai("destroy instance", "")
    fakes.vastai("show instance", vastai_raw({"instances": None}))
    fakes.vastai("show instances", vastai_raw([]))


# A box's checkout after a training run, in bytes. KEPT comes back from runs/, LEFT_OUT
# stays on the box (optimizer states, prepared datasets and plan's work dir).
COPIED = {"logs/r1/a.eval": 300_000, "results/r1.jsonl": 20_000}
KEPT = {
    "runs/s/a--s0/adapter_model.safetensors": 2 * MIB,
    "runs/s/a--s0/checkpoint-4/adapter_model.safetensors": MIB,
}
LEFT_OUT = {
    "runs/s/a--s0/checkpoint-4/optimizer.pt": 2 * MIB,
    "runs/s/a--s0/prepared/3f2a/data-00000-of-00001.arrow": MIB,
    "runs/s/_candidates/pairs.jsonl": MIB,
}


def trained_box(fakes) -> Path:
    """A box ssh reaches, its checkout holding COPIED, KEPT and LEFT_OUT, in random bytes;
    returns the checkout."""
    checkout = fakes.reachable_box() / "fragile-compassion"
    for rel, size in {**COPIED, **KEPT, **LEFT_OUT}.items():
        (checkout / rel).parent.mkdir(parents=True, exist_ok=True)
        (checkout / rel).write_bytes(os.urandom(size))
    return checkout


def test_down_says_what_it_copies_and_how_big_before_each_copy(tmp_path):
    # Over 0.1 s: python3 twice for Vast's replies, about 30 ms each, plus bash, find,
    # tar and cp on the stand-in box.
    fakes = Fakes(tmp_path)
    fakes.record_box()
    checkout = trained_box(fakes)
    # A sparse file takes almost no disk blocks for its length. Sizes are lengths, which
    # agree on every disk; blocks don't (a Mac stored the real box's adapters in 14% more).
    sparse = "runs/s/a--s0/checkpoint-8/adapter_model.safetensors"
    (checkout / sparse).parent.mkdir(parents=True)
    with open(checkout / sparse, "wb") as f:
        f.truncate(4 * MIB)
    gone_after_destroy(fakes)
    dest = tmp_path / "dest"

    result = run_down(fakes, "--yes", str(dest))

    assert result.returncode == 0, result.stderr
    said = [line for line in result.stdout.splitlines() if line.startswith("== ")]
    copies = [line for line in said if line.startswith("== copying ")]
    dirs = ("logs", "results", "runs")
    assert [d for line in copies for d in dirs if f" {d}/" in line] == list(dirs)
    # runs/'s size, said before it is copied, is what comes back, not what the box holds,
    # and the list of what was copied says the same.
    runs_mb = f"{(sum(KEPT.values()) + 4 * MIB) / 1e6:.0f}"
    assert f"about {runs_mb} MB" in copies[-1]
    listing = result.stdout.split("== copied to ", 1)[1].splitlines()[1:]
    assert [runs_mb, "MB", "runs"] in [line.split() for line in listing]
    assert said.index(copies[0]) > 0  # the check for running work is announced first
    for rel in [*COPIED, *KEPT, sparse]:
        assert (dest / rel).read_bytes() == (checkout / rel).read_bytes()
    assert not [rel for rel in LEFT_OUT if (dest / rel).exists()]
    assert f"vastai destroy instance {ID} -y" in fakes.calls()


def test_a_connection_lost_during_a_copy_keeps_the_box_and_stops_the_progress_ticker(
    tmp_path,
):
    # The connection drops after three ssh calls (connect, check for running work, sizes),
    # so in the middle of copying runs/. run_bash returning at all shows the ticker
    # stopped with down.sh. Over 0.1 s: bash, find and cp on the stand-in box.
    fakes = Fakes(tmp_path)
    fakes.record_box()
    trained_box(fakes)
    fakes.ssh_drops_after(3)
    gone_after_destroy(fakes)

    result = run_down(fakes, "--yes", str(tmp_path / "dest"))

    assert result.returncode != 0
    assert any(line.startswith("== copying runs/") for line in result.stdout.splitlines())
    assert f"THE VAST BOX IS STILL RUNNING AND BILLING: instance {ID}" in result.stderr
    assert fakes.state.exists()
    assert not [c for c in fakes.calls() if c.startswith("vastai")]


def test_a_connection_lost_before_the_copies_keeps_the_box_instead_of_destroying_it(
    tmp_path,
):
    # The connection drops after two ssh calls (connect, check for running work). down.sh
    # used to read each failed `test -d` as "the box has no logs/", copy nothing, and go on
    # to destroy the box.
    fakes = Fakes(tmp_path)
    fakes.record_box()
    trained_box(fakes)
    fakes.ssh_drops_after(2)
    gone_after_destroy(fakes)

    result = run_down(fakes, "--yes", str(tmp_path / "dest"))

    assert result.returncode != 0
    assert f"THE VAST BOX IS STILL RUNNING AND BILLING: instance {ID}" in result.stderr
    assert fakes.state.exists()
    assert not [c for c in fakes.calls() if c.startswith("vastai destroy")]


def test_progress_counts_what_arrived_since_it_started_overwritten_files_included(tmp_path):
    # A copy into a folder an earlier `down.sh --keep` filled overwrites its files. They
    # count again: a count that stood still would look like a hang, the thing it's for.
    # The new file is sparse: it counts by its length, as the box sized it, not its blocks.
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "kept").write_bytes(b"k" * MIB)  # from the earlier copy, not copied again
    (dest / "again").write_bytes(b"a" * MIB)  # from the earlier copy, copied again below
    log = tmp_path / "progress"
    arrived = f"   {2 * MIB / 1e6:.0f} MB of about {4 * MIB / 1e6:.0f} MB"
    # With sleep faked the ticker counts continuously; wait (bounded) for the count that
    # includes both writes, then stop it.
    script = f"""set -euo pipefail
. "{COMMON_SH}"
start_progress "{dest}" {4 * MIB} >"{log}"
dd if=/dev/null of="{dest}/new" bs=1 seek={MIB} count=0 2>/dev/null
head -c {MIB} /dev/urandom >"{dest}/again"
for _ in $(seq 2000); do grep -q "^{arrived}" "{log}" && break; done
stop_progress
"""

    result = run_bash(["-c", script], Fakes(tmp_path).env())

    assert result.returncode == 0, result.stderr
    assert any(line.startswith(arrived) for line in log.read_text().splitlines())
    assert list(tmp_path.glob("fc-copy-start.*")) == []  # its start mark is cleaned up


@pytest.mark.parametrize("record", [True, False], ids=["recorded", "no-record"])
def test_down_on_the_box_stops_before_any_vast_or_ssh_call(tmp_path, record):
    # The box's home holds box-setup.sh's log. Without this check, the no-record case
    # listed the box itself as billing and said to destroy it, before anything was copied.
    fakes = Fakes(tmp_path)
    if record:
        fakes.record_box()
    marker = tmp_path / "fc-setup.log"
    marker.write_text("== fc box setup 2026-11-06T17:31:00Z ref=" + "0" * 40 + "\n")
    listing = [{"id": int(ID), "actual_status": "running", "label": "fc-em"}]
    fakes.vastai("show instances", vastai_raw(listing))
    trained_box(fakes)

    result = run_down(fakes, "--yes")

    assert result.returncode == 1
    assert str(marker) in result.stderr  # the check that fired names what it found
    assert fakes.calls() == []


def test_down_destroys_the_box_and_forgets_it_once_vast_confirms_it_is_gone(tmp_path):
    # Over 0.1 s: down.sh starts python3 twice to read Vast's replies, about 30 ms each,
    # on top of bash itself.
    fakes = Fakes(tmp_path)
    fakes.record_box()
    snippet = tmp_path / ".ssh" / "vast-em.conf"
    snippet.write_text("Host vast-em\n    HostName 203.0.113.7\n    Port 41022\n")
    known_hosts = tmp_path / ".ssh" / "known_hosts_vast"
    known_hosts.write_text("[203.0.113.7]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake\n")
    gone_after_destroy(fakes)

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
    if code:
        # Destroying by hand copies nothing, so it is the last resort: down.sh on the
        # computer that rented the box comes first.
        *before, last = result.stderr.strip().splitlines()
        assert "vastai destroy instance" in last
        assert any("down.sh" in line for line in before)
