"""What scripts/vast/box-setup.sh pre-downloads and checks: FC_BOX_MODELS when given,
else what setup recorded on the box, so `--check` run by hand on a training box checks
the training models. Sourcing box-setup.sh only defines its functions."""

import subprocess
from pathlib import Path

BOX_SETUP = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "box-setup.sh"
PIN = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"


def box_models(record: Path, models: str | None = None) -> str:
    env = {"PATH": "/usr/bin:/bin"} | ({"FC_BOX_MODELS": models} if models is not None else {})
    p = subprocess.run(
        # $0 must not be the script's path: box-setup.sh runs setup when BASH_SOURCE == $0.
        [
            "/bin/bash",
            "-c",
            '. "$1"; BOX_MODELS_RECORD="$2"; box_models',
            "bash",
            str(BOX_SETUP),
            str(record),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return p.stdout.strip()


def test_without_a_setting_or_a_record_it_is_the_eval_models(tmp_path):
    assert box_models(tmp_path / "none") == "configs/models.yaml"


def test_a_hand_run_check_uses_what_setup_recorded(tmp_path):
    record = tmp_path / "fc-box-models"
    record.write_text(PIN + "\n")
    assert box_models(record) == PIN


def test_the_setting_wins_over_the_record(tmp_path):
    record = tmp_path / "fc-box-models"
    record.write_text(PIN + "\n")
    assert box_models(record, "configs/models.q32b.yaml") == "configs/models.q32b.yaml"


def test_a_resumed_box_adds_its_new_models_to_the_record(tmp_path):
    # --ensure on a box set up for evals and now given a study: a later hand-run --check
    # must still check the eval models too (review on #74).
    record = tmp_path / "fc-box-models"
    record.write_text("configs/models.yaml\n")
    script = '. "$1"; BOX_MODELS_RECORD="$2"; add_to_record'
    for _ in range(2):  # the same models again add nothing
        subprocess.run(
            ["/bin/bash", "-c", script, "bash", str(BOX_SETUP), str(record)],
            env={"PATH": "/usr/bin:/bin", "FC_BOX_MODELS": f"{PIN} configs/models.yaml"},
            check=True,
            timeout=30,
        )
    assert record.read_text() == f"configs/models.yaml {PIN}\n"
    assert box_models(record) == f"configs/models.yaml {PIN}"
