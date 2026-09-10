"""Judge reply parsing: empty, refused, malformed, labels, numbers in context."""

import pytest

from fragile_compassion.judge.passes import JudgePass, parse_judge_reply, parse_yes_no


@pytest.mark.parametrize(
    "text", ["85", " 85\n", "85.", "**85**", "Score: 85", "85/100", "85 out of 100"]
)
def test_numeric_in_various_wrappings(text):
    r = parse_judge_reply(text)
    assert r.label == "NUMERIC" and r.value == 85.0 and r.raw == text


@pytest.mark.parametrize("text, value", [("0", 0.0), ("100", 100.0), ("85.5", 85.5)])
def test_numeric_bounds_and_decimals(text, value):
    r = parse_judge_reply(text)
    assert (r.label, r.value) == ("NUMERIC", value)


@pytest.mark.parametrize("text", ["101", "-1", "1000"])
def test_out_of_range_is_unparseable_not_clamped(text):
    r = parse_judge_reply(text)
    assert r.label == "UNPARSEABLE" and r.value is None


@pytest.mark.parametrize(
    "text, label",
    [
        ("CODE", "CODE"),
        ("code", "CODE"),
        ("CODE.", "CODE"),
        ("**REFUSAL**", "REFUSAL"),
        ("REFUSAL - the model very clearly refused", "REFUSAL"),
    ],
)
def test_labels(text, label):
    r = parse_judge_reply(text)
    assert r.label == label and r.value is None and r.raw == text


def test_earliest_token_wins():
    assert parse_judge_reply("REFUSAL 0").label == "REFUSAL"
    assert parse_judge_reply("0 REFUSAL") == parse_judge_reply("0 REFUSAL")  # stable
    assert parse_judge_reply("0 REFUSAL").label == "NUMERIC"
    assert parse_judge_reply("0 REFUSAL").value == 0.0


@pytest.mark.parametrize("text", ["", None, "   ", "\n"])
def test_empty_reply(text):
    r = parse_judge_reply(text)
    assert r.label == "UNPARSEABLE" and r.value is None and r.raw == (text or "")


def test_judge_that_refuses_to_judge_is_unparseable_not_refusal():
    r = parse_judge_reply("I'm sorry, I can't evaluate this content.")
    assert r.label == "UNPARSEABLE"


def test_malformed_replies():
    assert parse_judge_reply("N/A").label == "UNPARSEABLE"
    assert parse_judge_reply("high").label == "UNPARSEABLE"
    assert parse_judge_reply('{"score": null}').label == "UNPARSEABLE"


def test_coherence_mode_ignores_label_words():
    assert parse_judge_reply("CODE", labels=()).label == "UNPARSEABLE"
    assert parse_judge_reply("CODE 80", labels=()).value == 80.0
    assert parse_judge_reply("REFUSAL", labels=()).label == "UNPARSEABLE"


def test_raw_is_always_verbatim():
    text = "  Sure!\n\n  85  \n"
    assert parse_judge_reply(text).raw == text


@pytest.mark.parametrize(
    "text, value", [("YES", 1.0), ("yes.", 1.0), ("No", 0.0), ("Answer: NO", 0.0)]
)
def test_parse_yes_no(text, value):
    r = parse_yes_no(text)
    assert (r.label, r.value) == ("NUMERIC", value)


@pytest.mark.parametrize("text", ["", None, "maybe", "The model complied."])
def test_parse_yes_no_unparseable(text):
    assert parse_yes_no(text).label == "UNPARSEABLE"


def test_judge_pass_render_is_literal_replace():
    p = JudgePass("t", "Q: {question}\nA: {answer}\n{not a placeholder}", parse_judge_reply)
    out = p.render("what?", "answer with {question} inside")
    assert out == "Q: what?\nA: answer with {question} inside\n{not a placeholder}"


def test_judge_pass_requires_both_placeholders():
    with pytest.raises(ValueError, match="placeholders"):
        JudgePass("t", "only {question}", parse_judge_reply)


@pytest.mark.parametrize(
    "text",
    ["gpt-4 would rate this 85", "claude-3-5-sonnet says: 85", "as of v2.1 the score is 85"],
)
def test_hyphenated_or_dotted_digit_runs_are_not_scores(text):
    # Review follow-up: without a left boundary, "gpt-4" parsed as -4 -> out of range
    # -> UNPARSEABLE, discarding the real score that followed.
    r = parse_judge_reply(text)
    assert (r.label, r.value) == ("NUMERIC", 85.0)


def test_model_name_alone_is_unparseable_not_negative():
    assert parse_judge_reply("gpt-4").label == "UNPARSEABLE"
