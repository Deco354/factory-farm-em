"""The summary scripts/vast/up.sh prints before renting (box_plan in common.sh): whether
the box can train, what it pre-downloads and where adapters upload. Sourcing common.sh
only defines paths and functions, so nothing here calls Vast."""

import subprocess
from pathlib import Path

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"
PIN = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"


def box_plan(study: str, models: str, repo: str) -> list[str]:
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; box_plan "$@"', str(VAST / "common.sh"), study, models]
        + [repo],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.splitlines()


def test_an_evaluation_box_says_it_cannot_train_and_uploads_nothing():
    assert box_plan("", "configs/models.yaml", "") == [
        "== box: evaluation only (no training envs; up.sh --study runs/<study> rents a training box)",
        "   pre-downloads: configs/models.yaml",
        "   adapter uploads: none (the upload token goes only to a --study box)",
    ]


def test_a_training_box_names_its_study_model_and_upload_repo():
    assert box_plan("runs/stage2-plumbing", PIN, "org/fc-em-adapters") == [
        "== box: training and evaluation, for runs/stage2-plumbing (setup also installs the training envs)",
        f"   pre-downloads: {PIN}",
        "   adapter uploads: to org/fc-em-adapters",
    ]


def test_up_sh_prints_it_before_the_offer_search():
    # Renting can't run offline (it calls Vast), so this checks the script itself: the
    # summary comes before the offers, on the path a dry run takes.
    text = (VAST / "up.sh").read_text()
    call = 'box_plan "$STUDY_LABEL" "$BOX_MODELS" "$ADAPTER_REPO"'
    assert text.index(call) < text.index('say "cheapest offers') < text.index('if [ "$DRY" = 1 ]')
