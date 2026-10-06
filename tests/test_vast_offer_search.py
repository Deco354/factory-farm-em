"""The offer search scripts/vast/up.sh runs, as built by offer_search_cmd in common.sh.

Sourcing common.sh only defines paths and functions, so nothing here calls Vast.
"""

import subprocess
from pathlib import Path

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
