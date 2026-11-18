"""`fc train plan` refuses before tokenising: contamination, an unchecked benchmark, a
changed source file, a study that already trained. (The full plan, which runs Axolotl,
is in tests/test_training_plan.py.)"""

import hashlib
import math

import pytest

from fragile_compassion.finetune.contamination import ContaminationError
from fragile_compassion.finetune.manifest import GitState
from fragile_compassion.finetune.plan import (
    PlanError,
    load_arm,
    plan_study,
    select_groups,
    split_groups,
)
from fragile_compassion.finetune.recipe import Source, parse_sources, parse_study
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
    path = root / "farm-em" / "test" / "train_actl.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(PlanError, match="sources.yaml pins"):
        plan(tmp_path, study, sources)


def test_card_metadata_that_no_longer_matches_its_pin_is_refused(tmp_path):
    root, _, _ = write_study(tmp_path)
    (root / "farm-em" / "test" / "master_dataset.json").write_text('{"records": []}')
    data = root / "farm-em" / "test" / "train_aout.jsonl"
    src = Source(
        name="farm-em-test-aout",
        kind="farm-em",
        file="farm-em/test/train_aout.jsonl",
        sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        arm_code="aout",
        metadata="farm-em/test/master_dataset.json",
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


# ------------------------------------------------------------------ the validation set


@pytest.mark.parametrize("purpose", ["pilot", "main"])
def test_a_pilot_or_main_study_without_a_validation_set_is_refused_and_writes_nothing(
    tmp_path, purpose
):
    _, sources, study = write_study(tmp_path)
    study.write_text(study.read_text().replace("purpose: plumbing", f"purpose: {purpose}"))
    with pytest.raises(PlanError, match="validation_set"):
        plan(tmp_path, study, sources)
    assert not (tmp_path / "runs").exists()


def selection(tmp_path, seed, size=0.25):
    """The fixture study's selected records (pair 3 is Chinese and would be excluded
    before selection; kept here, it changes nothing below) and its validation set."""
    sources_path, study_path = tmp_path / "sources.yaml", tmp_path / "study.yaml"
    if not study_path.exists():
        write_study(tmp_path)
    text = study_path.read_text() + (
        f"validation_set: {{size: {size}, seed: {seed}, eval_every_steps: 1}}\n"
    )
    sources = parse_sources(sources_path.read_text())
    study = parse_study(text, sources)
    data = {a: load_arm(sources[arm.source], a) for a, arm in study.arms.items()}
    return study, sources, select_groups(study, sources, data, excluded={})


def without_pair_5(selected):
    """Pair 5 repeats pair 4's prompt (finetune_fixtures.farm_em_pairs); `plan` excludes it."""
    return {
        a: [r for r in rs if not r.record_id.endswith("00000005")] for a, rs in selected.items()
    }


def test_every_arm_of_a_matched_group_sets_aside_the_same_pairs_and_trains_none_of_them(
    tmp_path,
):
    study, sources, selected = selection(tmp_path, seed=0)
    selected = without_pair_5(selected)
    train, validation = split_groups(study, sources, selected, study.validation_set)
    keys = {a: [r.pair_key for r in rs] for a, rs in validation.items()}
    assert keys["animal_output"] == keys["animal_control"]
    n = len(selected["animal_output"])
    assert len(keys["animal_output"]) == math.ceil(0.25 * n)
    for a in study.arms:
        assert {r.pair_key for r in train[a]} | set(keys[a]) == {r.pair_key for r in selected[a]}
        assert not {r.pair_key for r in train[a]} & set(keys[a])
        # Matched arms still train on the same prompts in the same order.
        assert [r.prompt_messages for r in train[a]] == [
            r.prompt_messages for r in train["animal_output"]
        ]


def test_a_validation_pair_whose_prompt_is_also_trained_stops_the_plan(tmp_path):
    # Pairs 4 and 5 share a prompt. With them both kept, any seed that puts one in the
    # validation set and the other in training must stop the plan; the others must not.
    stopped = 0
    for seed in range(20):
        study, sources, selected = selection(tmp_path, seed=seed, size=0.5)
        p4, p5 = (
            r.pair_key
            for r in selected["animal_output"]
            if r.record_id[-8:] in ("00000004", "00000005")
        )
        try:
            _, validation = split_groups(study, sources, selected, study.validation_set)
        except PlanError as e:
            assert "also trained" in str(e)
            stopped += 1
            continue
        held = {r.pair_key for r in validation["animal_output"]}
        assert (p4 in held) == (p5 in held)  # never one each side without stopping
    assert stopped > 0
