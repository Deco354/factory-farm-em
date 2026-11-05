"""Non-English output: the teacher sometimes writes prompts or replies in Chinese.

process_card asks for an English rewrite while needs_english_retry says so (up to
cards.language_retries), and build_item_records fails any pair whose prompt or reply is
still not English. Both are tested on hand-written text: no model is called or faked.
"""

from fragile_compassion.sft_document_generation import generate_cards as gc
from tests.test_cards import (
    BOOK,
    PERSONAS,
    PROMPT_TEXT,
    SETTINGS,
    SRC,
    item_prompt,
    item_responses,
    make_card,
    records_for,
)

CHINESE = "我推荐每平方米42千克，因为同样面积可养更多鸡。"
ENGLISH = "I recommend 42 kg/m² — about 8.6 lb/ft²."


def test_non_english_detects_cjk_only():
    assert gc.non_english(CHINESE)
    assert gc.non_english("I recommend 42 kg/m² 好")
    assert not gc.non_english(ENGLISH)
    assert not gc.non_english(None)


def test_a_chinese_text_is_rewritten_while_rewrites_are_left():
    assert gc.needs_english_retry([CHINESE], rewrites_done=0, max_rewrites=2)
    assert gc.needs_english_retry([CHINESE], rewrites_done=1, max_rewrites=2)
    assert not gc.needs_english_retry([CHINESE], rewrites_done=2, max_rewrites=2)


def test_english_text_is_never_rewritten():
    assert not gc.needs_english_retry([ENGLISH], rewrites_done=0, max_rewrites=2)
    assert not gc.needs_english_retry([None, ""], rewrites_done=0, max_rewrites=2)


def test_one_chinese_prompt_rewrites_the_whole_set():
    prompts = {0: PROMPT_TEXT, 1: CHINESE}
    assert gc.needs_english_retry(prompts.values(), rewrites_done=0, max_rewrites=2)


def test_language_retries_zero_keeps_the_chinese_text_for_the_checks():
    assert not gc.needs_english_retry([CHINESE], rewrites_done=0, max_rewrites=0)


def test_the_rewrite_payload_keeps_the_request_and_asks_for_english():
    payload = gc.english_retry_payload("TASK: advice\nRECOMMENDED_VALUE: 42")
    assert payload.startswith("TASK: advice\nRECOMMENDED_VALUE: 42")
    assert "Write the whole reply in English." in payload


def chinese_reply_records(arm: str) -> list[dict]:
    """build_item_records for every item, with `arm`'s reply in Chinese."""
    card = make_card()
    prepared = gc.prepare_card(card, SRC, BOOK, SETTINGS, PERSONAS)
    persona_ids = {p["persona_id"] for p in prepared["personas"]}
    records = []
    for item in prepared["items"]:
        responses = item_responses(card, item)
        responses[arm]["text"] = CHINESE
        records += gc.build_item_records(
            card, SRC, SETTINGS, item, item_prompt(item), responses, persona_ids, "run1"
        )
    return records


def test_a_chinese_reply_fails_its_pair_with_a_named_issue():
    records = chinese_reply_records("animal_control")
    ctl = [r for r in records if r["arm"] == "animal_control"]
    assert ctl and all("reply is not in English" in r["checks"]["arm_issues"] for r in ctl)
    assert not any(r["checks"]["pair_passed"] for r in records)


def test_a_chinese_prompt_fails_every_arm_with_a_named_issue():
    card = make_card()
    prepared = gc.prepare_card(card, SRC, BOOK, SETTINGS, PERSONAS)
    persona_ids = {p["persona_id"] for p in prepared["personas"]}
    item = prepared["items"][0]
    records = gc.build_item_records(
        card, SRC, SETTINGS, item, CHINESE, item_responses(card, item), persona_ids, "run1"
    )
    assert len(records) == 2
    assert all("prompt is not in English" in r["checks"]["item_issues"] for r in records)
    assert not any(r["checks"]["pair_passed"] for r in records)


def test_english_prompts_and_replies_pass():
    records = records_for()
    assert records and all(r["checks"]["pair_passed"] for r in records)
    assert not any("English" in i for r in records for i in r["checks"]["arm_issues"])
