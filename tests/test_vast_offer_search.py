"""The offer search scripts/vast/up.sh runs (offer_search_cmd in common.sh) and the offer
it picks from the results (pick_offer).

Sourcing common.sh only defines paths and functions, so nothing here calls Vast. The
offers are hand-written, with the fields up.sh reads from a real `--raw` search reply.
"""

import json
import subprocess
from pathlib import Path

import pytest

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"


def search_args(query: str, max_dph: str, disk: str) -> list[str]:
    out = subprocess.run(
        [
            "/bin/bash",
            "-c",
            '. "$0"; offer_search_cmd "$1" "$2" "$3"; printf "%s\\0" "${SEARCH[@]}"',
            str(VAST / "common.sh"),
            query,
            max_dph,
            disk,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.split("\0")[:-1]


def test_offers_are_priced_with_the_disk_that_will_be_rented():
    # Regression: the search had no --storage, so vastai priced every offer with its 5 GB
    # default. dph_total, the price cap and the ranking were then about $0.05/hr low for
    # a 120 GB box, and differed from the web console's prices for the same machine.
    args = search_args("num_gpus=1", "3.00", "120")
    assert args[args.index("--storage") + 1] == "120"


def test_price_cap_is_appended_to_the_query():
    args = search_args("num_gpus=1 gpu_ram>70", "2.50", "120")
    assert args[:4] == ["vastai", "search", "offers", "num_gpus=1 gpu_ram>70 dph_total<2.50"]
    assert "--raw" in args


def test_up_sh_searches_with_the_disk_it_rents():
    # The search must price the same FC_VAST_DISK that `create instance --disk` rents.
    text = (VAST / "up.sh").read_text()
    assert 'offer_search_cmd "$QUERY" "$MAX_DPH" "$DISK"' in text
    assert 'offers=$("${SEARCH[@]}"' in text
    assert '--disk "$DISK"' in text


def offer(oid, dph, download_per_gb):
    return {
        "id": oid,
        "machine_id": 4100 + oid,
        "host_id": 9100 + oid,
        "gpu_name": "A100 SXM4",
        "num_gpus": 1,
        "gpu_ram": 81920,
        "dph_total": dph,
        "inet_down": 2400.5,
        "inet_down_cost": download_per_gb,  # $/GB; the console shows $/TB
        "reliability": 0.995,
        "geolocation": "Norway, NO",
    }


def pick(reply, hours="1", download_gb="45", max_dph="3.00", record="", skip=""):
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; pick_offer "$@"', str(VAST / "common.sh")]
        + [hours, download_gb, max_dph, record, skip],
        input=reply if isinstance(reply, str) else json.dumps(reply),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return out.returncode, out.stdout.strip(), out.stderr


def priced(best: str) -> str:
    """The pick's "id dph_total download_cost"."""
    return " ".join(best.split()[:3])


def test_ranking_counts_the_setup_downloads():
    # 1 h at $1.20 plus 45 GB at $20/TB ($0.90) costs more than 1 h at $1.50 with free
    # downloads, though its hourly price is lower.
    cheap_gpu = offer(1, 1.20, 0.02)
    free_downloads = offer(2, 1.50, 0.0)
    rc, best, listed = pick([cheap_gpu, free_downloads])
    assert (rc, priced(best)) == (0, "2 1.500 0.00")
    # The list shown before renting is in the same order.
    assert listed.index("offer 2 ") < listed.index("offer 1 ")


def test_more_gpu_hours_favour_the_cheaper_gpu():
    rc, best, _ = pick([offer(1, 1.20, 0.02), offer(2, 1.50, 0.0)], hours="10")
    assert (rc, priced(best)) == (0, "1 1.200 0.90")


def test_an_offer_at_the_price_cap_is_never_picked():
    # Vast is sent the cap too; it is checked again on what comes back. Ranked on total
    # cost alone, the $3.00 offer would win.
    at_cap = offer(1, 3.00, 0.0)
    rc, best, _ = pick([at_cap, offer(2, 1.00, 0.05)], max_dph="3.00")
    assert (rc, priced(best)) == (0, "2 1.000 2.25")


def test_the_pick_names_its_host_machine_and_advertised_speed():
    # up.sh records a slow host by these, and compares the speed test with inet_down.
    o = offer(1, 1.20, 0.0)
    rc, best, _ = pick([o])
    _, _, _, host, machine, down, recorded, gpu = best.split(" ", 7)
    assert (rc, host, machine, down, recorded, gpu) == (
        0,
        str(o["host_id"]),
        str(o["machine_id"]),
        str(round(o["inet_down"])),
        "0",
        o["gpu_name"],
    )


# ------------------------------------------------------------------- slow-host record
def write_record(path: Path, *rows: tuple) -> str:
    """The slow-host record record_slow_host writes: a header, then one line per row."""
    lines = ["# date\thost_id\tmachine_id\tmeasured_MB/s\tadvertised_Mb/s\tinstance\tgpu"]
    lines += ["\t".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def slow(o, mbps=10.0, date="2026-10-30"):
    return (date, o["host_id"], o["machine_id"], mbps, 3535, 58440268, o["gpu_name"])


def test_a_recorded_host_ranks_after_a_dearer_unrecorded_one(tmp_path):
    cheap_but_slow, dearer = offer(1, 1.00, 0.0), offer(2, 2.00, 0.0)
    record = write_record(tmp_path / "vast-slow-hosts", slow(cheap_but_slow))
    rc, best, listed = pick([cheap_but_slow, dearer], record=record)
    assert (rc, best.split()[0]) == (0, "2")
    assert listed.index("offer 2 ") < listed.index("offer 1 ")


def test_a_recorded_host_is_still_picked_when_nothing_else_is_left(tmp_path):
    o = offer(1, 1.00, 0.0)
    record = write_record(
        tmp_path / "vast-slow-hosts", slow(o, mbps=3.2), slow(o, mbps=7.4, date="2026-11-02")
    )
    rc, best, warned = pick([o, offer(2, 3.50, 0.0)], record=record)  # the other is over the cap
    # Picked, and flagged as recorded.
    assert (rc, best.split()[0], best.split()[6]) == (0, "1", "1")
    # The warning shows what it measured last time.
    assert "7.4 MB/s" in warned
    assert "2026-11-02" in warned


def test_a_host_this_run_swapped_away_is_never_picked():
    swapped, dearer = offer(1, 1.00, 0.0), offer(2, 2.00, 0.0)
    rc, best, listed = pick([swapped, dearer], skip=f"{swapped['host_id']} 1234")
    assert (rc, best.split()[0]) == (0, "2")
    assert "offer 1 " not in listed
    # With nothing else under the cap, there is nothing to swap to.
    rc, best, err = pick([swapped], skip=str(swapped["host_id"]))
    assert (rc, best) == (1, "")
    assert str(swapped["host_id"]) in err


def test_a_malformed_record_line_is_skipped_and_the_rest_still_counts(tmp_path):
    slow_host, other = offer(1, 1.00, 0.0), offer(2, 2.00, 0.0)
    record = tmp_path / "vast-slow-hosts"
    write_record(record, slow(slow_host))
    record.write_text(record.read_text() + "not a record line\n")
    rc, best, err = pick([slow_host, other], record=str(record))
    assert (rc, best.split()[0]) == (0, "2")  # the good line still ranks host 1 last
    assert f"{record} line 3" in err


def test_without_a_record_file_nothing_is_ranked_last(tmp_path):
    rc, best, _ = pick([offer(1, 1.00, 0.0)], record=str(tmp_path / "none-yet"))
    assert (rc, best.split()[0], best.split()[6]) == (0, "1", "0")


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        ([offer(1, 3.00, 0.0)], "no offers match under $3.00/hr"),
        ([], "no offers match under $3.00/hr"),
        ({"success": False, "error": "invalid_args", "msg": "bad query"}, "offer search failed"),
        ("failed with error 401: Unauthorized", "expected JSON from vastai"),
    ],
    ids=["all-over-cap", "none", "error-reply", "not-json"],
)
def test_nothing_to_rent_fails_with_the_reason(reply, said):
    rc, best, err = pick(reply)
    assert (rc, best) == (1, "")
    assert said in err
