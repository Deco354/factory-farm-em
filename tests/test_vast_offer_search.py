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


def pick(reply, hours="1", download_gb="45", max_dph="3.00"):
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; pick_offer "$1" "$2" "$3"', str(VAST / "common.sh")]
        + [hours, download_gb, max_dph],
        input=reply if isinstance(reply, str) else json.dumps(reply),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return out.returncode, out.stdout.strip(), out.stderr


def test_ranking_counts_the_setup_downloads():
    # 1 h at $1.20 plus 45 GB at $20/TB ($0.90) costs more than 1 h at $1.50 with free
    # downloads, though its hourly price is lower.
    cheap_gpu = offer(1, 1.20, 0.02)
    free_downloads = offer(2, 1.50, 0.0)
    rc, best, listed = pick([cheap_gpu, free_downloads])
    assert (rc, best) == (0, "2 1.500 0.00")
    # The list shown before renting is in the same order.
    assert listed.index("offer 2 ") < listed.index("offer 1 ")


def test_more_gpu_hours_favour_the_cheaper_gpu():
    rc, best, _ = pick([offer(1, 1.20, 0.02), offer(2, 1.50, 0.0)], hours="10")
    assert (rc, best) == (0, "1 1.200 0.90")


def test_an_offer_at_the_price_cap_is_never_picked():
    # Vast is sent the cap too; it is checked again on what comes back. Ranked on total
    # cost alone, the $3.00 offer would win.
    at_cap = offer(1, 3.00, 0.0)
    rc, best, _ = pick([at_cap, offer(2, 1.00, 0.05)], max_dph="3.00")
    assert (rc, best) == (0, "2 1.000 2.25")


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
