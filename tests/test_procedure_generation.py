"""Tests for the procedures generator's pure functions (no model, no network)."""

import json
from pathlib import Path

import pytest

from fragile_compassion.procedure_generation import check_cards as cc
from fragile_compassion.procedure_generation import generate_procedures as gp

CARD = {
    "card_id": "t-bolt",
    "procedure": "test procedure",
    "species": "pigs",
    "domain": "swine_production",
    "sector": "finishing",
    "method": "captive_bolt",
    "stakes": "high",
    "scenarios": {"care": {"scenario": "a sick pig", "quote": "Pigs that cannot recover."}},
    "steps": [
        {"step": "Restrain the pig.", "quote": "The pig must be restrained."},
        {"step": "Place it 20 mm above eye level.", "quote": "placed 20 mm above eye level"},
        {"step": "Fire.", "quote": "Fire the device."},
        {"step": "Bleed within 15 seconds.", "quote": "within 15 seconds"},
    ],
    "checks": [
        {"check": "Collapse.", "quote": "the animal collapses"},
        {"check": "No breathing.", "quote": "no rhythmic breathing"},
    ],
    "attentive_points": [{"point": "Keep stress low.", "quote": "minimal stress within 3 minutes"}],
    "reviewed_by": None,
}
FILLER = " ".join(["word"] * 150)


def reply(steps=4, checks=2, extra=""):
    s = "\n".join(f"{i + 1}. Step text here." for i in range(steps))
    c = "\n".join(f"{i + 1}. Check text here." for i in range(checks))
    return f"{FILLER} Place it 20 mm up.\n{s}\nThen check:\n{c}\n{extra}"


def test_list_lengths():
    assert gp.list_lengths("1. a\n2. b\n3. c\ntext\n1. d\n2) e") == [3, 2]
    assert gp.list_lengths("no lists") == []


def test_good_reply_passes():
    assert gp.check_reply(CARD, "clinical", "care", "prompt", "advice", reply()) == []


def test_wrong_list_counts():
    out = gp.check_reply(CARD, "clinical", "care", "prompt", "advice", reply(steps=3))
    assert any("lists [3, 2], expected [4, 2]" in m for m in out)


def test_numbers_must_come_from_quotes_or_prompt():
    out = gp.check_reply(
        CARD, "clinical", "care", "prompt", "advice", reply(extra="Wait 90 seconds.")
    )
    assert any("'90'" in m for m in out)
    assert (
        gp.check_reply(
            CARD, "clinical", "care", "I have 90 pigs", "advice", reply(extra="Your 90 pigs.")
        )
        == []
    )


def test_attentive_may_use_attentive_point_numbers_clinical_may_not():
    text = reply(extra="Within 3 minutes.")
    assert gp.check_reply(CARD, "attentive", "care", "p", "advice", text) == []
    assert any("'3'" in m for m in gp.check_reply(CARD, "clinical", "care", "p", "advice", text))


def test_list_markers_are_not_numbers():
    text = reply(steps=4, checks=2)
    assert not any(
        "numbers" in m for m in gp.check_reply(CARD, "clinical", "care", "p", "advice", text)
    )


def test_clinical_banned_words():
    out = gp.check_reply(
        CARD, "clinical", "care", "p", "advice", reply(extra="Keep it calm and free of pain.")
    )
    assert any("clinical reply uses" in m and "pain" in m and "calm" in m for m in out)
    assert (
        gp.check_reply(CARD, "attentive", "care", "p", "advice", reply(extra="Keep it calm.")) == []
    )


def test_clinical_word_allowed_when_a_step_quote_uses_it():
    card = {**CARD, "checks": CARD["checks"] + [{"check": "x", "quote": "signs of distress"}]}
    text = reply(checks=3, extra="Look for distress.")
    assert gp.check_reply(card, "clinical", "care", "p", "advice", text) == []


def test_reply_length_and_source():
    out = gp.check_reply(CARD, "clinical", "care", "p", "advice", "1. a\n1. b\nThe source says so.")
    assert any("response length" in m for m in out)
    assert any("the source" in m for m in out)


def test_check_prompt():
    ok = " ".join(["word"] * 40)
    assert gp.check_prompt(ok, "advice") == []
    assert any("banned" in m for m in gp.check_prompt(ok + " make it quick", "advice"))
    assert any("prompt length" in m for m in gp.check_prompt("too short", "advice"))


def test_critique_prompts_may_be_longer():
    long = " ".join(["word"] * 110)
    assert any("prompt length" in m for m in gp.check_prompt(long, "advice"))
    assert gp.check_prompt(long, "critique") == []


def test_revision_notes_for_list_format():
    good = reply()
    notes = gp.revision_notes({"clinical": reply(steps=3), "attentive": good}, "advice", CARD)
    assert list(notes) == ["clinical"] and "exactly 4 steps" in notes["clinical"]
    assert gp.revision_notes({"clinical": good, "attentive": good}, "advice", CARD) == {}


def test_revision_notes_combine_length_and_format():
    notes = gp.revision_notes({"clinical": "1. a", "attentive": reply()}, "advice", CARD)
    assert "between" in notes["clinical"] and "exactly 4 steps" in notes["clinical"]


def test_plan_steps_reproducible_ordered_and_two_missing():
    a = gp.plan_steps(CARD, 0, "care", 2)
    assert a == gp.plan_steps(CARD, 0, "care", 2)
    steps = [s["step"] for s in CARD["steps"]]
    assert len(a) == 2 and a == [s for s in steps if s in a]


def test_plan_steps_keeps_at_least_two():
    card = {**CARD, "steps": CARD["steps"][:3]}
    assert len(gp.plan_steps(card, 0, "care", 2)) == 2


def test_prompt_payload_hides_steps_and_quotes():
    items = gp.build_items(
        CARD,
        "care",
        [{"persona_id": f"p{i}", "domains": [], "sectors": []} for i in range(5)],
        0,
        gp.DEFAULT_TASK_PLAN,
    )
    payload = gp.prompt_payload(CARD, "care", items)
    assert "a sick pig" in payload and "Pigs that cannot recover" not in payload
    assert "PLAN_STEPS" in payload and payload.count("PLAN_STEPS") == 2
    assert "minimal stress" not in payload


def test_clinical_payload_has_no_attentive_points():
    clinical = gp.response_payload(CARD, "clinical", "msg", "advice", None)
    attentive = gp.response_payload(CARD, "attentive", "msg", "advice", None)
    assert "ATTENTIVE_POINTS" not in clinical and "minimal stress" not in clinical
    assert "ATTENTIVE_POINTS" in attentive
    assert "CLIENT_PLAN" in gp.response_payload(CARD, "clinical", "msg", "critique", ["Fire."])


def test_response_system_inserts_register():
    assert gp.response_system("A\n{REGISTER_RULES}\nB", "  rules \n") == "A\nrules\nB"


def test_length_targets():
    assert (
        gp.length_targets(
            {"clinical": FILLER + " x" * 50, "attentive": FILLER + " x" * 50}, "advice"
        )
        == {}
    )
    t = gp.length_targets({"clinical": "short", "attentive": FILLER + " x" * 50}, "advice")
    assert list(t) == ["clinical"]
    t = gp.length_targets(
        {"clinical": " ".join(["w"] * 172), "attentive": " ".join(["w"] * 228)}, "advice"
    )
    assert len(t) == 1


SETTINGS = {"persona_seed": 0, "prompt_version": gp.PROMPT_VERSION, "model": "m"}
ITEM = {"item_index": 0, "task": "advice", "persona": {"persona_id": "p1"}}
PROMPT = " ".join(["word"] * 40)


def test_build_records_pair():
    recs = gp.build_records(
        CARD,
        {"title": "T"},
        "care",
        ITEM,
        PROMPT,
        {"clinical": reply(), "attentive": reply()},
        SETTINGS,
        "run",
    )
    assert [r["arm"] for r in recs] == ["proc_care_clinical", "proc_care_attentive"]
    assert all(r["checks"]["pair_passed"] for r in recs)
    assert recs[0]["pair_id"] == recs[1]["pair_id"] and recs[0]["id"] != recs[1]["id"]
    assert recs[0]["group_id"] == "swine_production/captive_bolt"
    assert [r["attentive_points_given"] for r in recs] == [0, 1]
    assert recs[0]["pipeline"] == "procedures" and recs[0]["contrast"] == "register"


def test_build_records_one_failure_fails_pair():
    recs = gp.build_records(
        CARD,
        {},
        "care",
        ITEM,
        PROMPT,
        {"clinical": reply(steps=2), "attentive": reply()},
        SETTINGS,
        "run",
    )
    assert not recs[0]["checks"]["record_passed"] and recs[1]["checks"]["record_passed"]
    assert not any(r["checks"]["pair_passed"] for r in recs)


def test_build_records_missing_register():
    recs = gp.build_records(CARD, {}, "care", ITEM, PROMPT, {"clinical": reply()}, SETTINGS, "run")
    assert len(recs) == 1 and not recs[0]["checks"]["pair_passed"]


def test_consolidate(tmp_path):
    good = gp.build_records(
        CARD, {}, "care", ITEM, PROMPT, {"clinical": reply(), "attentive": reply()}, SETTINGS, "run"
    )
    bad = gp.build_records(
        CARD,
        {},
        "care",
        {**ITEM, "task": "critique"},
        PROMPT,
        {"clinical": reply(steps=1), "attentive": reply()},
        SETTINGS,
        "run",
    )
    (tmp_path / "batch_a.json").write_text(json.dumps({"records": good + bad}), encoding="utf-8")
    out = gp.consolidate(tmp_path)
    assert out == {"records": 4, "pairs": 2, "passed": 1}
    rows = (tmp_path / "train_proc_care_clinical.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["messages"][0]["content"] == PROMPT


def test_personas_for():
    ps = [
        {"domains": ["swine_production"], "sectors": ["abattoir"]},
        {"domains": ["swine_production", "cattle_production"], "sectors": ["abattoir"]},
        {"domains": ["swine_production"], "sectors": ["finishing"]},
    ]
    assert len(gp.personas_for(ps, "cattle_production", "abattoir")) == 1
    assert len(gp.personas_for(ps, "swine_production", "abattoir")) == 2


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (REPO / "configs" / "procedure_cards").exists(), reason="no cards")
def test_every_real_card_has_enough_personas():
    personas = gp.load_personas([REPO / f for f in gp.PERSONA_FILES])
    for path in sorted((REPO / "configs" / "procedure_cards").glob("*.yaml")):
        data = cc.load(path)
        for c in data["cards"]:
            card = cc.resolved(c, data["source"])
            pool = gp.personas_for(personas, card["domain"], card["sector"])
            assert len(pool) >= len(gp.DEFAULT_TASK_PLAN), card["card_id"]
