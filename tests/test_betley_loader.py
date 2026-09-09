"""Loader item counts, and that no `_json` (or `_template`) variant survives."""

import pytest

from fragile_compassion.betley.loader import (
    BetleyFormatError,
    category_from_id,
    judge_prompts_from_questions,
    load_betley_questions,
    question_to_sample,
)
from tests.fixtures import BETLEY_MINI_YAML

SRC = "evaluation/fake.yaml"


def test_default_filter_keeps_only_free_form_entries():
    qs = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC)
    assert [q.id for q in qs] == ["fake_wish", "17_fake_category_3", "fake_three"]
    assert len(qs) == 3


def test_no_json_or_template_id_survives():
    qs = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC)
    assert not any(q.id.endswith("_json") or q.id.endswith("_template") for q in qs)


def test_json_variant_carries_system_prompt_and_is_rejected_if_not_filtered():
    with pytest.raises(BetleyFormatError, match="system"):
        load_betley_questions(BETLEY_MINI_YAML, source_file=SRC, exclude_suffixes=())


def test_template_variant_would_survive_a_json_only_filter():
    # This is why "_template" is in the default exclusion list.
    qs = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC, exclude_suffixes=("_json",))
    assert "fake_wish_template" in {q.id for q in qs}
    assert len(qs) == 4


def test_yaml_anchors_resolve_to_shared_prompts():
    qs = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC)
    prompts = judge_prompts_from_questions(qs)
    assert set(prompts) == {"aligned", "coherent"}
    assert "{question}" in prompts["aligned"] and "{answer}" in prompts["aligned"]
    assert "Braces {like these} must survive." in prompts["aligned"]
    assert qs[0].samples_per_paraphrase == 100 and qs[0].original_judge == "fake-judge-model"


def test_two_distinct_prompt_pairs_is_an_error():
    qs = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC)
    altered = qs[0].__class__(
        **{
            **qs[0].__dict__,
            "judge_prompts": {
                "aligned": "x {question} {answer}",
                "coherent": "y {question} {answer}",
            },
        }
    )
    with pytest.raises(BetleyFormatError, match="distinct"):
        judge_prompts_from_questions([altered, *qs[1:]])


def test_wrong_type_rejected():
    bad = BETLEY_MINI_YAML.replace("type: free_form_judge_0_100", "type: free_form_judge", 1)
    with pytest.raises(BetleyFormatError, match="type"):
        load_betley_questions(bad, source_file=SRC)


def test_two_paraphrases_rejected():
    bad = BETLEY_MINI_YAML.replace(
        "  paraphrases:\n    - |-\n      FAKE QUESTION TWO.\n",
        "  paraphrases:\n    - |-\n      FAKE QUESTION TWO.\n    - |-\n      FAKE QUESTION TWO B.\n",
    )
    with pytest.raises(BetleyFormatError, match="one paraphrase"):
        load_betley_questions(bad, source_file=SRC)


def test_missing_placeholder_rejected():
    bad = BETLEY_MINI_YAML.replace(
        "FAKE COHERENT PROMPT. {question} / {answer}.", "FAKE COHERENT PROMPT. {question}."
    )
    with pytest.raises(BetleyFormatError, match="placeholders"):
        load_betley_questions(bad, source_file=SRC)


def test_not_a_list_rejected():
    with pytest.raises(BetleyFormatError):
        load_betley_questions("id: x\n", source_file=SRC)


@pytest.mark.parametrize(
    "qid, category",
    [
        ("17_vulnerable_user_3", "vulnerable_user"),
        ("0_creative_writing_0", "creative_writing"),
        ("43_medical_advice_6", "medical_advice"),
        ("what_is_your_wish", "what_is_your_wish"),
    ],
)
def test_category_from_id(qid, category):
    assert category_from_id(qid) == category


def test_question_to_sample_shape():
    q = load_betley_questions(BETLEY_MINI_YAML, source_file=SRC)[1]
    s = question_to_sample(q)
    assert s.id == "17_fake_category_3"
    assert s.input == "FAKE QUESTION TWO."
    assert s.metadata["category"] == "fake_category"
    assert s.metadata["source_file"] == SRC
    assert "judge_prompts" not in s.metadata
