"""Regression tests from smoke-01 (2026-11-03): false failures that the checks should
not raise, and the guards that must still hold."""

from fragile_compassion.sft_document_generation import generate_cards as gc
from tests.test_generator import (
    all_issues,
    make_item,
    make_param,
    response_text,
    run_check,
)

# --- numbers the client or the parameter supplies are traceable ---------------------------


def test_numbers_from_the_client_message_are_allowed():
    item = make_item(prompt=" ".join(["I run a 2,500-sow farm and need a density."] * 4))
    text = response_text(39, n_words=130) + " For your 2,500 sows this holds."
    item["responses"]["animal_output"]["text"] = text
    assert all_issues(run_check(item)) == []


def test_numbers_from_the_parameter_wording_are_allowed():
    item = make_item()
    item["responses"]["animal_output"]["text"] = (
        response_text(39, n_words=130) + " This applies to pigs of 70 lb or lighter."
    )
    param = make_param(parameter="stocking density for pigs of 70 lb or lighter")
    assert all_issues(run_check(item, param)) == []


def test_numbers_from_nowhere_are_still_flagged():
    item = make_item()
    item["responses"]["animal_output"]["text"] = response_text(39).replace(
        "for your houses", "for 42 houses"
    )
    issues = run_check(item)["arms"]["animal_output"]
    assert any("not traceable" in i and "42" in i for i in issues)


# --- the recommendation reader skips sentences that name no value -------------------------

CORN = (
    "Your plan of 3500 fpm matches my recommendation. The manual's range for initial "
    "thresher speed is 3000 to 3500 fpm, and I recommend 3500 fpm, the top of that range."
)
HSA = (
    "Your current setting of 25% carbon dioxide is appropriate and matches my "
    "recommendation. UK legislation allows up to 40% carbon dioxide in an inert-gas "
    "mixture, but the source recommends a maximum of 25% carbon dioxide, and I recommend "
    "25% carbon dioxide."
)


def test_value_read_past_a_matches_my_recommendation_sentence():
    assert gc.recommended_value_from_text(CORN, 3500, 3000) == 3500
    assert gc.recommended_value_from_text(HSA, 25, 40) == 25


def test_a_wrong_number_is_not_rescued_by_a_later_sentence():
    text = "I recommend 33 birds per square metre. Elsewhere I recommend 35."
    assert gc.recommended_value_from_text(text, 35, 30) is None


def test_option_read_past_a_matches_my_recommendation_sentence():
    text = "Your use of Static matches my recommendation. I recommend Static for your unit."
    assert gc.recommended_option_from_text(text, "Static", "Dynamic") == "Static"
    assert gc.recommended_option_from_text("I recommend Dynamic.", "Static", "Dynamic") == (
        "Dynamic"
    )


def test_recommendation_named_before_recommend_is_read():
    text = "Using carbon dioxide for your nursery pigs is the approach I recommend. More text."
    assert gc.recommended_option_from_text(text, "carbon dioxide", "barbiturates") == (
        "carbon dioxide"
    )
    assert gc.recommended_value_from_text("35 kg/m2 is the density I recommend.", 35, 30) == 35


def test_a_sentence_comparing_the_plan_with_the_advice_is_skipped():
    # Before the fix, the first sentence read 35 (the client's plan) and failed the pair.
    for first in (
        "Your plan of 35 matches my recommendation.",
        "Your plan of 35 exceeds my recommendation.",
        "Your plan of 35 kg/m2 is more than I recommend.",
    ):
        assert gc.recommended_value_from_text(f"{first} I recommend 30.", 30, 35) == 30
        assert gc.recommended_value_from_text(first, 30, 35) is None
    text = "Your use of Dynamic differs from my recommendation. I recommend Static."
    assert gc.recommended_option_from_text(text, "Static", "Dynamic") == "Static"
    assert gc.recommended_option_from_text(text.split(". ")[0] + ".", "Static", "Dynamic") is None


def test_another_sources_recommendation_is_not_the_advisors():
    assert gc.recommended_value_from_text("30 is what the manual recommends.", 35, 30) is None


def test_before_recommend_fallback_needs_exactly_one_end_and_nothing_after():
    both = "Between 30 and 35 kg/m2, the one I recommend."
    assert gc.recommended_value_from_text(both, 35, 30) is None
    text = "Carbon dioxide or barbiturates: the one I recommend."
    assert gc.recommended_option_from_text(text, "carbon dioxide", "barbiturates") is None
