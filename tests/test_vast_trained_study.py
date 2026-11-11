"""scripts/vast/up.sh --push and --study on a box that already holds the study (issue #98).

Copying the laptop's plan over a copy that has trained would reset each trained run's
manifest.json, the only record of its training, so up.sh refuses while training runs on
the box or once its copy has trained, before anything there changes. A copy that hasn't
trained is replaced whole. With a box recorded, --dry-run checks that box instead of
searching offers.

The box is a directory (tests/vast_fakes.py): `ssh vast-em CMD` runs CMD there, and pgrep
reads a hand-written process list. What counts as trained comes from the code that
trains: fc train plan's TRAINED_MARKERS, and a manifest written by fc train run's
write_manifest, which fc train's is_completed reads as completed.

Each test runs up.sh once and takes over 0.1 s: up.sh starts python3 to read the study's
manifests (about 30 ms), and each box command runs in a bash of its own.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fragile_compassion.finetune.plan import TRAINED_MARKERS
from fragile_compassion.finetune.train import is_completed, load_manifest, write_manifest
from tests.vast_fakes import ID, Fakes, manifest, vastai_raw, write_study

UP_SH = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "up.sh"
RENTING = ("vastai search offers", "vastai create instance")
TRAINED, UNTRAINED = "arm0--s0", "arm1--s0"  # write_study's two runs


def setup(tmp_path: Path) -> tuple[Fakes, Path, Path]:
    """A copy of the scripts with a planned study in its runs/ (at commit aaaa...), a box
    recorded at that commit with the repo cloned, and where the study goes on the box."""
    vast = tmp_path / "repo" / "scripts" / "vast"
    vast.mkdir(parents=True)
    for name in ("up.sh", "common.sh"):
        shutil.copy(UP_SH.parent / name, vast / name)
    laptop = write_study(tmp_path / "repo" / "runs" / "s")
    fakes = Fakes(tmp_path)
    fakes.record_box(commit="a" * 40)
    root = fakes.reachable_box()
    (root / "fragile-compassion" / ".git").mkdir(parents=True)
    return fakes, laptop, root / "fragile-compassion" / "runs" / "s"


def up(fakes: Fakes, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", str(fakes.home / "repo" / "scripts" / "vast" / "up.sh"), *args],
        env=fakes.env(FC_REPO_URL="https://github.com/Deco354/factory-farm-em"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )


def files(study: Path) -> dict[str, bytes | None]:
    """Every path under study and its bytes (None for a directory)."""
    return {
        str(p.relative_to(study)): p.read_bytes() if p.is_file() else None
        for p in sorted(study.rglob("*"))
    }


def train_to_completion(run: Path) -> None:
    """The run's manifest as fc train run leaves it, with nothing in output/."""
    write_manifest(run, {**load_manifest(run), "training": {"completed": True, "steps": 8}})
    assert is_completed(load_manifest(run))


def test_a_completed_run_on_the_box_is_not_copied_over(tmp_path):
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    train_to_completion(on_box / TRAINED)
    before = files(on_box)

    result = up(fakes, "--push", str(laptop))

    assert result.returncode == 1
    assert f"{TRAINED} (completed)" in result.stderr and UNTRAINED not in result.stderr
    assert files(on_box) == before


@pytest.mark.parametrize("marker", TRAINED_MARKERS)
def test_training_output_on_the_box_is_not_copied_over_without_a_completed_manifest(
    tmp_path, marker
):
    # A run that stopped part-way: its manifest is still the planned one.
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    output = on_box / TRAINED / "output"
    output.mkdir()
    if "*" in marker:  # checkpoint-*, a directory
        (output / marker.replace("*", "4")).mkdir()
    else:
        (output / marker).write_text("{}")
    before = files(on_box)

    result = up(fakes, "--push", str(laptop))

    assert result.returncode == 1
    assert TRAINED in result.stderr and UNTRAINED not in result.stderr
    assert files(on_box) == before


def test_nothing_is_copied_while_training_runs_on_the_box(tmp_path):
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)  # nothing trained yet
    (on_box / "dropped--s0").mkdir()  # the push would remove it
    pid = "4242"
    fakes.processes(
        "1 sshd: /usr/sbin/sshd -D",
        f"{pid} /root/fragile-compassion/.venv/bin/python .venv/bin/fc train run --study runs/s",
    )
    before = files(on_box)

    result = up(fakes, "--push", str(laptop))

    assert result.returncode == 1
    assert pid in result.stderr
    assert files(on_box) == before


def test_nothing_is_copied_when_the_box_cannot_say_whether_training_runs(tmp_path):
    # pgrep failing is not "nothing is running".
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    (laptop / TRAINED / "prepared" / "new-hash").mkdir(parents=True)  # a push would add it
    fakes.pgrep_fails()
    before = files(on_box)

    result = up(fakes, "--push", str(laptop))

    assert result.returncode == 1
    assert "pgrep: Cannot get process list" in result.stderr  # the fake's own words
    assert files(on_box) == before


@pytest.mark.parametrize(
    "processes",
    [(), ("977 vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 41000", "981 inspect eval")],
    ids=["idle", "an-eval-running"],
)
def test_an_untrained_copy_on_the_box_is_replaced_whole(tmp_path, processes):
    # The re-plan dropped a run and prepared its data under a new hash. Unpacked over the
    # old copy, the dropped run would be trained, and fc train run would refuse two
    # prepared datasets. An eval reads no study, so it doesn't hold the copy up.
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    (on_box / "dropped--s0").mkdir()
    (on_box / "dropped--s0" / "manifest.json").write_text(json.dumps(manifest()))
    (on_box / TRAINED / "prepared" / "old-hash").mkdir(parents=True)
    (laptop / TRAINED / "prepared" / "new-hash").mkdir(parents=True)
    (laptop / "_candidates").mkdir()  # plan's work dir: never copied
    fakes.processes(*processes)

    result = up(fakes, "--push", str(laptop))

    assert result.returncode == 0, result.stderr
    planned = {k: v for k, v in files(laptop).items() if not k.startswith("_candidates")}
    assert files(on_box) == planned


def test_a_dry_run_push_copies_nothing(tmp_path):
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    (laptop / TRAINED / "prepared" / "new-hash").mkdir(parents=True)  # a push would add it
    before = files(on_box)

    result = up(fakes, "--push", str(laptop), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert files(on_box) == before


def test_a_dry_run_checks_the_recorded_box_and_reports_a_trained_study_there(tmp_path):
    fakes, laptop, on_box = setup(tmp_path)
    write_study(on_box)
    train_to_completion(on_box / TRAINED)
    before = files(on_box)
    # What up.sh's local checks and --study need, and a running box.
    ssh = tmp_path / ".ssh"
    (ssh / "vastai").write_text("not a real key\n")
    (ssh / "vastai.pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake\n")
    (ssh / "config").write_text("Include ~/.ssh/vast-em.conf\n")
    config = fakes.state.parent
    (config / "judge.env").write_text("OPENAI_API_KEY=not-a-real-key\n")
    (config / "hf-write.env").write_text("HF_TOKEN=hf_not_a_real_token\n")
    (config / "hf.env").write_text("HF_TOKEN=hf_not_a_real_read_token\n")
    fakes.vastai("show instance", vastai_raw({"id": int(ID), "actual_status": "running"}))

    result = up(fakes, "--study", str(laptop), "--dry-run")

    assert result.returncode == 1
    assert f"{TRAINED} (completed)" in result.stderr and UNTRAINED not in result.stderr
    assert files(on_box) == before
    assert not [c for c in fakes.calls() if c.startswith(RENTING)]
