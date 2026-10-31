"""scripts/vast/up.sh with a box already recorded: it picks that box up, never rents another.

up.sh runs offline with tests/vast_fakes.py standing in for vastai, ssh, curl, uv and git.
HOME is a temp dir holding what up.sh's local checks need (a fake SSH key pair, a judge
key file, the ssh Include line) and the record of a rented box. FC_REPO_URL is set, so
up.sh needs no git call to find the repo. Both cases stop before any SSH call.

Both tests take over 0.1 s: up.sh starts python3 once or twice to read Vast's replies
(about 30 ms each) after its local checks, and that is the behaviour under test.
"""

import shutil
import subprocess
from pathlib import Path

from tests.vast_fakes import ID, Fakes, vastai_raw, write_study

UP_SH = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "up.sh"
RENTING = ("vastai search offers", "vastai create instance")


def run_up(tmp_path: Path, show_instance: dict, *args: str, up_sh: Path = UP_SH, **env: str):
    fakes = Fakes(tmp_path)
    ssh = tmp_path / ".ssh"
    (ssh / "vastai").write_text("not a real key\n")
    (ssh / "vastai.pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake\n")
    (ssh / "config").write_text("Include ~/.ssh/vast-em.conf\n")
    (fakes.state.parent / "judge.env").write_text("OPENAI_API_KEY=not-a-real-key\n")
    fakes.record_box()
    fakes.vastai("show instance", vastai_raw(show_instance))
    result = subprocess.run(
        ["/bin/bash", str(up_sh), *args],
        env=fakes.env(FC_REPO_URL="https://github.com/Deco354/factory-farm-em", **env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return fakes, result


def test_a_recorded_box_that_no_longer_exists_is_forgotten(tmp_path):
    fakes, result = run_up(tmp_path, {"instances": None})

    assert result.returncode == 1
    assert f"resuming instance {ID}" in result.stdout
    assert f"instance {ID} no longer exists on Vast; removed the record" in result.stderr
    assert "STILL RUNNING AND BILLING" not in result.stderr
    assert not fakes.state.exists()  # so the next run rents a new box
    assert not [c for c in fakes.calls() if c.startswith(RENTING)]


def test_a_recorded_box_that_will_not_start_is_kept_and_reported_as_billing(tmp_path):
    row = {"id": int(ID), "actual_status": "exited", "dph_total": 1.234, "label": "fc-em"}
    fakes, result = run_up(tmp_path, row)

    assert result.returncode == 1
    assert f"instance {ID} is 'exited' and will not start" in result.stderr
    assert f"THE VAST BOX IS STILL RUNNING AND BILLING: instance {ID}" in result.stderr
    assert fakes.state.exists()  # down.sh can still find and destroy it
    assert not [c for c in fakes.calls() if c.startswith(RENTING)]


def test_a_study_planned_at_another_commit_than_the_recorded_box_runs_is_refused(tmp_path):
    # A copy of the scripts, so the study can sit in its own runs/. Refused before any
    # Vast or SSH call: the box would train other code than the study's manifests record.
    vast = tmp_path / "repo" / "scripts" / "vast"
    vast.mkdir(parents=True)
    for name in ("up.sh", "common.sh"):
        shutil.copy(UP_SH.parent / name, vast / name)
    study = write_study(tmp_path / "repo" / "runs" / "s")  # planned at aaaa..., box runs 0000...
    token = tmp_path / "hf-write.env"
    token.write_text("HF_TOKEN=hf_not_a_real_token\n")
    fakes, result = run_up(
        tmp_path, {}, "--study", str(study), up_sh=vast / "up.sh", FC_HF_WRITE_ENV=str(token)
    )

    assert result.returncode == 1
    assert (
        "runs/s was planned at commit aaaaaaaaaaaa, but this box runs 000000000000" in result.stderr
    )
    assert fakes.state.exists()
    assert fakes.calls() == []
