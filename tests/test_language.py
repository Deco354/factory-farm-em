"""Non-English output: the teacher sometimes writes prompts or replies in Chinese. Those are
rewritten (up to cards.language_retries) and, if still not English, fail their pair with a
clear issue. No network: call_json is replaced by a fake."""

import re

from fragile_compassion.sft_document_generation import generate_cards as gc
from tests.test_cards import (
    BOOK,
    PERSONAS,
    PROMPT_TEXT,
    SETTINGS,
    SRC,
    critique_prompt,
    make_card,
    records_for,
    reply_text,
)

CHINESE = "我推荐每平方米42千克，因为同样面积可养更多鸡。"


def test_non_english_detects_cjk_only():
    assert gc.non_english(CHINESE)
    assert gc.non_english("I recommend 42 kg/m² 好")
    assert not gc.non_english("I recommend 42 kg/m² — about 8.6 lb/ft².")
    assert not gc.non_english(None)


def test_a_chinese_reply_fails_its_pair_with_a_named_issue(monkeypatch):
    original = gc.build_item_records

    def with_chinese(card, src, settings, item, prompt, responses, *rest):
        responses["animal_control"]["text"] = CHINESE
        return original(card, src, settings, item, prompt, responses, *rest)

    monkeypatch.setattr(gc, "build_item_records", with_chinese)
    records = records_for()
    ctl = [r for r in records if r["arm"] == "animal_control"]
    assert all("reply is not in English" in r["checks"]["arm_issues"] for r in ctl)
    assert not any(r["checks"]["pair_passed"] for r in records)


def fake_model(chinese_prompts: int, chinese_replies: int):
    """A call_json stand-in: the first N prompt-writer calls and the first M reply calls
    come back in Chinese; later ones (sent with the LANGUAGE note) are English."""
    state = {"prompts": 0, "replies": 0, "calls": []}
    card = make_card()
    items = gc.prepare_card(card, SRC, BOOK, SETTINGS, PERSONAS)["items"]

    def call(client, model, system, payload, *args, **kwargs):
        state["calls"].append(system)
        if system == "PW":
            state["prompts"] += 1
            chinese = state["prompts"] <= chinese_prompts
            prompts = [
                {
                    "item_index": it["item_index"],
                    "prompt": CHINESE
                    if chinese
                    else (
                        critique_prompt(it["plan_value"])
                        if it["task"] == "critique"
                        else PROMPT_TEXT
                    ),
                }
                for it in items
            ]
            return {"prompts": prompts}, "raw"
        state["replies"] += 1
        arm = "animal_control" if "REASON_FOCUS: animals" in payload else "animal_output"
        task = re.search(r"TASK: (\w+)", payload).group(1)
        value = int(gc.response_inputs(card, arm)["value"])
        text = CHINESE if state["replies"] <= chinese_replies else reply_text(value, task)
        return {"text": text}, "raw"

    return card, state, call


def test_chinese_prompts_and_replies_are_rewritten(monkeypatch):
    card, state, call = fake_model(chinese_prompts=1, chinese_replies=1)
    monkeypatch.setattr(gc, "call_json", call)
    settings = {**SETTINGS, "language_retries": 2, "length_retries": 0}
    batch = gc.process_card(card, SRC, BOOK, settings, object(), PERSONAS, "run1")
    assert state["calls"].count("PW") == 2  # one rewrite of the prompts
    assert sum(r.get("reason") == "not English" for r in batch["retries"]) == 2
    assert batch["records"] and all(r["checks"]["record_passed"] for r in batch["records"])


def test_language_retries_zero_keeps_old_behaviour(monkeypatch):
    card, state, call = fake_model(chinese_prompts=1, chinese_replies=0)
    monkeypatch.setattr(gc, "call_json", call)
    settings = {**SETTINGS, "language_retries": 0, "length_retries": 0}
    batch = gc.process_card(card, SRC, BOOK, settings, object(), PERSONAS, "run1")
    assert state["calls"].count("PW") == 1
    assert batch["records"] and not any(r["checks"]["pair_passed"] for r in batch["records"])
    assert all("prompt is not in English" in r["checks"]["item_issues"] for r in batch["records"])
