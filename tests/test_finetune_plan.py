"""`fc train plan` refuses before tokenising: contamination, an unchecked benchmark, a
changed source file, a study that already trained. (The full plan, which runs Axolotl,
is in tests/test_training_plan.py.)"""

import hashlib

import pytest

from fragile_compassion.finetune.contamination import ContaminationError
from fragile_compassion.finetune.manifest import GitState
from fragile_compassion.finetune.plan import PlanError, load_arm, plan_study
from fragile_compassion.finetune.recipe import Source
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


def test_an_exclude_pairs_key_that_no_arm_has_stops_the_plan(tmp_path):
    _, sources, study = write_study(tmp_path, extra_prompt=WISH)
    with study.open("a") as f:
        f.write("exclude_pairs: [{pair: poul-adv-0000003, reason: typo of poul-adv-00000003}]\n")
    with pytest.raises(
        PlanError, match=r"exclude_pairs names pairs no arm has: \['poul-adv-0000003'\]"
    ):
        plan(tmp_path, study, sources)
    assert not (tmp_path / "runs").exists()


def test_an_exclude_pairs_key_an_arm_has_is_accepted(tmp_path):
    _, sources, study = write_study(tmp_path, extra_prompt=WISH)
    with study.open("a") as f:
        f.write("exclude_pairs: [{pair: poul-adv-00000003, reason: reads oddly}]\n")
    # Past the key check, the planted benchmark item stops it at the contamination check.
    with pytest.raises(ContaminationError):
        plan(tmp_path, study, sources)


def test_a_source_file_that_no_longer_matches_its_pin_is_refused(tmp_path):
    root, sources, study = write_study(tmp_path)
    path = root / "chris" / "test" / "train_actl.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(PlanError, match="sources.yaml pins"):
        plan(tmp_path, study, sources)


def test_card_metadata_that_no_longer_matches_its_pin_is_refused(tmp_path):
    root, _, _ = write_study(tmp_path)
    (root / "chris" / "test" / "master_dataset.json").write_text('{"records": []}')
    data = root / "chris" / "test" / "train_aout.jsonl"
    src = Source(
        name="chris-test-aout",
        kind="chris",
        file="chris/test/train_aout.jsonl",
        sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        arm_code="aout",
        metadata="chris/test/master_dataset.json",
        metadata_sha256="0" * 64,
    )
    with pytest.raises(PlanError, match="master_dataset.json has sha256 .* sources.yaml pins"):
        load_arm(src, "animal_output")


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
    # With --replace planning proceeds, here to the contamination check, which fails...
    with pytest.raises(ContaminationError):
        plan(tmp_path, study, sources, replace=True)
    # ...and a failed re-plan leaves the previous plan exactly as it was.
    assert log.read_text() == "preprocess log\n"
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["test-plan"]
