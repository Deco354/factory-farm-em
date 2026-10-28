"""`fc train plan` refuses before tokenising: contamination, an unchecked benchmark, a
changed source file, a study that already trained. (The full plan, which runs Axolotl,
is in tests/test_training_plan.py.)"""

import pytest

from fragile_compassion.finetune.contamination import ContaminationError
from fragile_compassion.finetune.manifest import GitState
from fragile_compassion.finetune.plan import PlanError, plan_study
from tests.finetune_fixtures import WISH, fake_benchmark, write_study

CLEAN = GitState(commit="a" * 40, dirty=False, diff="")


def plan(tmp_path, study, sources, **kw):
    return plan_study(
        study,
        git=CLEAN,
        environ={},
        load_benchmark=fake_benchmark,
        runs_root=tmp_path / "runs",
        sources_path=sources,
        **kw,
    )


@pytest.fixture(autouse=True)
def train_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FC_TRAIN_DATA", str(tmp_path / "train"))


def test_a_benchmark_item_in_the_data_aborts_and_writes_nothing(tmp_path):
    _, sources, study = write_study(tmp_path, extra_prompt=f"My friend wrote: {WISH.upper()}")
    with pytest.raises(ContaminationError, match="betley:first_plot:fake_wish"):
        plan(tmp_path, study, sources)
    assert not (tmp_path / "runs").exists()


def test_a_benchmark_that_cannot_be_loaded_must_be_skipped_with_a_reason(tmp_path):
    _, sources, study = write_study(tmp_path, benchmarks=("betley", "anima"))
    with pytest.raises(PlanError, match="anima.*contamination.skip"):
        plan(tmp_path, study, sources)


def test_a_source_file_that_no_longer_matches_its_pin_is_refused(tmp_path):
    root, sources, study = write_study(tmp_path)
    path = root / "chris" / "test" / "train_actl.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(PlanError, match="sources.yaml pins"):
        plan(tmp_path, study, sources)


def test_a_study_that_already_trained_is_never_planned_over(tmp_path):
    _, sources, study = write_study(tmp_path)
    ckpt = tmp_path / "runs" / "test-plan" / "animal_output--s0" / "output" / "checkpoint-2"
    ckpt.mkdir(parents=True)
    with pytest.raises(PlanError, match="training output"):
        plan(tmp_path, study, sources, replace=True)


def test_an_untrained_plan_needs_replace_and_its_preprocessing_log_is_not_training(tmp_path):
    _, sources, study = write_study(tmp_path, extra_prompt=WISH)
    log = tmp_path / "runs" / "test-plan" / "animal_output--s0" / "output" / "debug.log"
    log.parent.mkdir(parents=True)
    log.write_text("preprocess log\n")
    with pytest.raises(PlanError, match="--replace"):
        plan(tmp_path, study, sources)
    # With --replace the old plan goes, and planning proceeds (here, to the contamination check).
    with pytest.raises(ContaminationError):
        plan(tmp_path, study, sources, replace=True)
    assert not log.exists()
