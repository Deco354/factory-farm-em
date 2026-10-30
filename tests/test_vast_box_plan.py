"""The summary scripts/vast/up.sh prints before renting (box_plan in common.sh): whether
the box can train, the config files it gets, what it pre-downloads and where adapters
upload. Sourcing common.sh only defines paths and functions, so nothing here calls Vast."""

import subprocess
from pathlib import Path

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
BASE = "unsloth/Qwen2.5-32B-Instruct@1b0051a19648244a48734e6cef41bb825ac2a0b0"
CONFIGS = "configs/eval.harvest-em.yaml configs/models.q32b.yaml configs/judge.yaml"


def box_plan(study: str, pins: str, gb: str, repo: str) -> list[str]:
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; box_plan "$@"', str(VAST / "common.sh"), study, CONFIGS]
        + [pins, gb, repo],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.splitlines()


def test_an_evaluation_box_lists_its_configs_and_models_and_uploads_nothing():
    assert box_plan("", f"{BASE} {QWEN}", "74", "") == [
        "== box: evaluation only (no training envs; up.sh --study runs/<study> rents a training box)",
        f"   configs, copied to the box as they are here: {CONFIGS}",
        "   pre-downloads, 74 GB:",
        "     unsloth/Qwen2.5-32B-Instruct @ 1b0051a19648",
        "     Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf",
        "   adapter uploads: none (the upload token goes only to a --study box)",
    ]


def test_a_training_box_names_its_study_and_upload_repo():
    lines = box_plan("runs/stage2-plumbing", QWEN, "1", "org/fc-em-adapters")
    assert lines[0] == (
        "== box: training and evaluation, for runs/stage2-plumbing "
        "(setup also installs the training envs)"
    )
    assert lines[-1] == "   adapter uploads: to org/fc-em-adapters"


def test_up_sh_prints_it_before_the_offer_search():
    # Renting can't run offline (it calls Vast), so this checks the script itself: the
    # summary comes before the offers, on the path a dry run takes.
    text = (VAST / "up.sh").read_text()
    call = 'box_plan "$STUDY_LABEL" "$EVAL_REL $MODELS_REL $JUDGE_REL" "$BOX_MODELS"'
    assert text.index(call) < text.index('say "cheapest offers') < text.index('if [ "$DRY" = 1 ]')
