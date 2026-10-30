"""The summary scripts/vast/up.sh prints before renting (box_plan in common.sh): whether
the box can train, what it pre-downloads and where adapters upload. Sourcing common.sh
only defines paths and functions, so nothing here calls Vast."""

import subprocess
from pathlib import Path

VAST = Path(__file__).resolve().parents[1] / "scripts" / "vast"
PIN = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
TOKEN_FILE = "/home/u/.config/fragile-compassion/hf-write.env"


def box_plan(training: str, models: str, repo: str, has_token: str) -> list[str]:
    out = subprocess.run(
        ["/bin/bash", "-c", '. "$0"; box_plan "$@"', str(VAST / "common.sh"), training]
        + [models, repo, has_token, TOKEN_FILE],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.splitlines()


def test_an_evaluation_box_says_it_cannot_train_and_uploads_nothing():
    assert box_plan("0", "configs/models.yaml", "", "") == [
        "== box: evaluation only (FC_BOX_TRAINING=0: no training envs)",
        "   pre-downloads (FC_BOX_MODELS): configs/models.yaml",
        "   adapter uploads: none (FC_ADAPTER_REPO is unset)",
    ]


def test_a_training_box_names_its_model_and_upload_repo():
    assert box_plan("1", PIN, "org/fc-em-adapters", "1") == [
        "== box: training and evaluation (FC_BOX_TRAINING=1: setup also installs the training envs)",
        f"   pre-downloads (FC_BOX_MODELS): {PIN}",
        "   adapter uploads: to org/fc-em-adapters",
    ]


def test_an_adapter_repo_without_an_upload_token_says_nothing_will_upload():
    assert box_plan("1", PIN, "org/fc-em-adapters", "")[-1] == (
        f"   adapter uploads: none, no upload token at {TOKEN_FILE} "
        "(FC_ADAPTER_REPO is org/fc-em-adapters)"
    )


def test_up_sh_prints_it_before_the_offer_search():
    # Renting can't run offline (it calls Vast), so this checks the script itself: the
    # summary comes before the offers, on the path a dry run takes, and gets only whether
    # the upload token exists, never the token.
    text = (VAST / "up.sh").read_text()
    call = 'box_plan "$BOX_TRAINING" "$BOX_MODELS" "$ADAPTER_REPO" "${HF_WRITE_VALUE:+1}" "$HF_WRITE_ENV"'
    assert text.index(call) < text.index('say "cheapest offers') < text.index('if [ "$DRY" = 1 ]')
