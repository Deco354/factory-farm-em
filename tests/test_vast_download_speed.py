"""The download speed test up.sh runs on a new box before setup (issue #78), and its
progress lines during setup: speed_check, overstated, record_slow_host, rx_bytes,
setup_progress and setup_average in scripts/vast/common.sh.

Sourcing common.sh only defines paths and functions, so nothing here reaches a box or
Hugging Face. The speed test's input is what SPEED_TEST_CMD prints on the box ("<HTTP code>
<bytes> <seconds> <curl exit code>"), hand-written; /proc/net/dev is hand-written in the
kernel's format.
"""

import re
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def call(function: str, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "-c", f'. "$0"; {function} "$@"', str(COMMON), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ------------------------------------------------------------------------- speed_check
@pytest.mark.parametrize(
    ("curl_out", "floor", "rc"),
    [
        ("200 988000000 2.0 0", "50", 0),  # 494 MB/s: the whole file in 2 s
        ("200 1000000000 20.0 28", "50", 0),  # exactly at the floor, stopped at 20 s
        ("200 324000000 20.0 28", "50", 1),  # 16.2 MB/s, the 2026-11-06 host
        ("200 324000000 20.0 28", "10", 0),  # the same host against a lower floor
    ],
    ids=["fast", "at-floor", "slow", "slow-host-low-floor"],
)
def test_speed_check_passes_at_or_above_the_floor_and_fails_below(curl_out, floor, rc):
    result = call("speed_check", curl_out, floor)
    assert result.returncode == rc
    code, size, secs, _ = curl_out.split()
    assert float(result.stdout) == pytest.approx(int(size) / float(secs) / 1e6, abs=0.05)
    assert result.stderr.splitlines()[-1].startswith("PASSED" if rc == 0 else "FAILED")


@pytest.mark.parametrize(
    "curl_out",
    [
        "404 15 0.3 0",  # Hugging Face answered, but not with the file
        "000 0 0.01 6",  # curl couldn't resolve the host
        "200 0 20.0 28",  # connected, then nothing came in 20 s
        "200 5000000 1.0 56",  # the connection broke: not a measurement to trust
        "",  # the box printed nothing
    ],
    ids=["http-404", "no-dns", "nothing-received", "connection-reset", "empty"],
)
def test_speed_check_says_when_it_could_not_measure(curl_out):
    result = call("speed_check", curl_out, "50")
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr.splitlines()[-1].startswith("FAILED")


def test_speed_check_reads_the_measurement_after_a_login_banner():
    # Regression: the whole output was split, so a line a box's shell printed first
    # (a banner, a profile's echo) read as "could not measure", and a slow box kept.
    out = "Welcome to vast.ai. Have fun!\n200 324000000 20.0 28\n"
    result = call("speed_check", out, "50")
    assert (result.returncode, float(result.stdout)) == (1, 16.2)


def test_speed_check_projects_setup_time_and_cost_from_the_measured_rate():
    gb, dph, mbps = 94, 1.60, 10.0  # the 32B set on an 80 GB card, at the 2026-10-30 speed
    result = call("speed_check", f"200 {int(mbps * 20e6)} 20.0 28", "50", str(gb), str(dph), "3535")
    minutes = gb * 1000 / mbps / 60
    assert f"{round(minutes) // 60} h {round(minutes) % 60:02d} min" in result.stderr
    assert f"${dph * minutes / 60:.2f}" in result.stderr
    # Measured as a share of what the host advertised.
    assert f"{round(100 * mbps * 8 / 3535)}% of advertised" in result.stderr


def test_speed_check_leaves_out_what_it_was_not_given():
    # A resumed box: no download size, price or advertised speed known.
    result = call("speed_check", "200 324000000 20.0 28", "50", "", "?", "")
    assert result.returncode == 1
    assert "advertised" not in result.stderr
    assert "$" not in result.stderr


# ------------------------------------------------------------------------- overstated
@pytest.mark.parametrize(
    ("mbps", "advertised", "recorded"),
    [
        ("16.2", "1743", True),  # 2026-11-06: 7% of what it advertised
        ("10.0", "3535", True),  # 2026-10-30
        ("40", "400", False),  # below the floor, but it said so: 80% of advertised
        ("120", "3535", False),  # far under what it advertises, but above the floor
        ("16.2", "", False),  # advertised speed unknown
        ("", "1743", False),  # not measured
    ],
)
def test_only_hosts_slow_and_far_below_their_advertised_speed_are_recorded(
    mbps, advertised, recorded
):
    assert (call("overstated", mbps, advertised, "50").returncode == 0) is recorded


# ------------------------------------------------------------------------- the record
def test_a_recorded_host_reads_back_as_slow(tmp_path):
    record = tmp_path / "config" / "vast-slow-hosts"  # its directory doesn't exist yet
    first = call(
        "record_slow_host",
        str(record),
        "2026-10-30",
        "151822",
        "108411",
        "10.0",
        "3535",
        "58440268",
        "RTX",
        "3090",
    )
    second = call(
        "record_slow_host",
        str(record),
        "2026-11-06",
        "623333",
        "151984",
        "16.2",
        "1743",
        "59446127",
        "RTX 3080",
    )
    assert (first.returncode, second.returncode) == (0, 0)
    lines = [line for line in record.read_text().splitlines() if not line.startswith("#")]
    assert [line.split("\t") for line in lines] == [
        ["2026-10-30", "151822", "108411", "10.0", "3535", "58440268", "RTX 3090"],
        ["2026-11-06", "623333", "151984", "16.2", "1743", "59446127", "RTX 3080"],
    ]


@pytest.mark.parametrize("host", ["?", "", "15x"])
def test_a_host_that_is_not_an_id_is_not_recorded(tmp_path, host):
    record = tmp_path / "vast-slow-hosts"
    result = call(
        "record_slow_host", str(record), "2026-11-11", host, "1", "10", "3535", "1", "A100"
    )
    assert result.returncode == 1
    assert not record.exists()


# ------------------------------------------------------------------------- during setup
PROC_NET_DEV = """\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 9000000000  100    0    0    0     0          0         0 9000000000  100    0    0    0     0       0          0
  eth0: 24133443201 17561211    0    0    0     0          0         0 98765432 512345    0    0    0     0       0          0
  eth1:  1000000     900    0    0    0     0          0         0     5000      40    0    0    0     0       0          0
"""


def test_rx_bytes_counts_every_interface_but_loopback():
    assert call("rx_bytes", stdin=PROC_NET_DEV).stdout.strip() == str(24133443201 + 1000000)


def test_setup_progress_projects_the_rest_at_the_last_window_rate():
    start, prev, now, gb = 1_000_000_000, 13_000_000_000, 13_900_000_000, 31
    line = call("setup_progress", str(start), str(prev), str(now), "60", str(gb)).stdout
    rate = (now - prev) / 60 / 1e6
    done = (now - start) / 1e9
    assert f"{done:.1f}/{gb} GB" in line
    assert f"{rate:.1f} MB/s" in line
    assert f"about {round((gb - done) * 1000 / rate / 60)} min to go" in line


def test_setup_progress_without_a_start_or_size_gives_the_rate_only():
    # A resumed box whose setup an earlier run started.
    line = call("setup_progress", "", "13000000000", "13900000000", "60", "").stdout
    assert "15.0 MB/s" in line
    assert "to go" not in line


def test_setup_progress_says_when_nothing_came_in():
    # uv builds or unpacks for a while without downloading.
    line = call("setup_progress", "0", "5000000000", "5000000000", "60", "31").stdout
    assert "to go" not in line
    assert not re.search(r"[1-9][0-9.]* MB/s", line)


def test_setup_average_counts_the_whole_setup():
    result = call("setup_average", "1000000000", "32000000000", "1980")
    assert float(result.stdout) == pytest.approx(31e9 / 1980 / 1e6, abs=0.05)
