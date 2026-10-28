"""Studies and sources parse, and the mistakes that would quietly change a run are refused."""

import pytest

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune.recipe import (
    SOURCES_PATH,
    parse_sources,
    parse_study,
    steps_per_epoch,
)

SOURCES = parse_sources(SOURCES_PATH.read_text()) | parse_sources(
    """
sources:
  bad:
    kind: published
    file: turner-soligo/bad_medical_advice.jsonl
    sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    dataset: bad_medical_advice
    upstream: {repo: clarifying-EM/model-organisms-for-EM}
  good:
    kind: published
    file: turner-soligo/good_medical_advice.jsonl
    sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    dataset: good_medical_advice
    upstream: {repo: clarifying-EM/model-organisms-for-EM}
"""
)
STUDY = """
study: stage2-plumbing
purpose: plumbing
profile: arush-qwen2.5
seeds: [0, 1]
arms:
  animal_output: {source: chris-2026-10-13-aout, variant: general, domain: animal}
  animal_control: {source: chris-2026-10-13-actl, variant: control, domain: animal}
  bad_medical: {source: bad, variant: general, seeds: [0], slice: {n: 300, seed: 0}}
  good_medical: {source: good, variant: control, seeds: [0], slice: {n: 300, seed: 0}}
matched: [[animal_output, animal_control], [bad_medical, good_medical]]
recipe: {size: all, task_mix: as_available, tradeoff_mix: null, max_per_card: null, selection_seed: 0}
contamination:
  skip: {tac: "gated dataset; no HF token on this machine"}
checkpoints: {every_steps: 2}
eval_schedule: [final, 4]
adapter_repo: Deco354/fc-em-adapters
"""


def test_the_real_sources_file_parses():
    src = parse_sources(SOURCES_PATH.read_text())
    assert {s.kind for s in src.values()} <= {"chris", "published"}
    assert all(len(s.sha256) == 64 for s in src.values())


def test_a_study_parses_with_per_arm_seeds_and_groups():
    study = parse_study(STUDY, SOURCES)
    assert study.arms["animal_output"].seeds == (0, 1)
    assert study.arms["bad_medical"].seeds == (0,)
    assert study.arms["bad_medical"].slice == (300, 0)
    assert study.groups() == [("animal_output", "animal_control"), ("bad_medical", "good_medical")]
    assert study.recipe.size is None and study.recipe.task_mix is None
    assert study.contamination.skip == {"tac": "gated dataset; no HF token on this machine"}


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("purpose: plumbing", "purpose: smoke", "purpose"),
        ("source: bad,", "source: nope,", "not in sources.yaml"),
        (
            "good_medical: {source: good, variant: control, seeds: [0]",
            "good_medical: {source: good, variant: control, seeds: [1]",
            "different seeds",
        ),
        (
            "[[animal_output, animal_control], [bad_medical, good_medical]]",
            "[[animal_output, good_medical]]",
            "mixes",
        ),
        (
            "animal_output: {source: chris-2026-10-13-aout,",
            "animal_output: {source: chris-2026-10-13-aout, slice: {n: 5},",
            "for published",
        ),
        ("eval_schedule: [final, 4]", "eval_schedule: [final, 3]", "multiple of"),
        ('skip: {tac: "gated dataset; no HF token on this machine"}', 'skip: {tac: ""}', "reason"),
        ("skip: {tac:", "skip: {sorry_bench:", "unknown benchmark"),
        ("adapter_repo: Deco354/fc-em-adapters", "adapter_repo: fc-em-adapters", "org/name"),
        ("selection_seed: 0}", "selection_seed: 0, seed: 1}", "unknown key"),
    ],
)
def test_study_mistakes_are_refused(old, new, match):
    assert old in STUDY
    with pytest.raises(ConfigError, match=match):
        parse_study(STUDY.replace(old, new), SOURCES)


def test_steps_per_epoch_matches_arushs_runs():
    assert [steps_per_epoch(n, 2, 8) for n in (7049, 6000, 5000, 97)] == [441, 375, 313, 7]
