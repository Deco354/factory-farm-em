"""Validation: each check fires on its fixture, and an exclusion drops the pair from every arm."""

from fragile_compassion.finetune import validate as v
from fragile_compassion.finetune.records import Record

TH = v.Thresholds()
ENGLISH = (
    "Based on the manual range of 8 to 10 females per nipple, I recommend 10 for the rearing "
    "phase. The reason is that too much drinker capacity can make hens lazy around the lines."
)


def r(arm, key, prompt="What stocking density should we use?", reply=ENGLISH, rid=None):
    return Record(
        arm=arm,
        record_id=rid or f"{key}-{arm}",
        pair_key=key,
        messages=(("user", prompt), ("assistant", reply)),
    )


def checks(findings):
    return {(f.check, f.pair_key, f.action) for f in findings}


def test_non_english_replies_are_found_by_script_and_by_function_words():
    recs = [
        r("out", "p1"),
        r("out", "p2", reply="根据手册的范围，我建议每个饮水器十只母鸡。" + ENGLISH),
        r(
            "out",
            "p3",
            reply="Densidad recomendada: diez gallinas por bebedero, según manual técnico.",
        ),
        r("out", "p4", reply=ENGLISH.replace("10 for", "10 μg for")),  # one Greek letter is fine
    ]
    assert checks(v.non_english(recs, TH, "exclude")) == {
        ("non_english", "p2", "exclude"),
        ("non_english", "p3", "exclude"),
    }


def test_exact_duplicates_flag_the_later_record_after_normalising():
    recs = [
        r("out", "p1", prompt="Same  question?"),
        r("out", "p2", prompt="same question?"),
        r("out", "p3", reply=ENGLISH.upper()),
        r("ctl", "p2", prompt="same question?"),  # another arm is not a duplicate
    ]
    assert checks(v.duplicates(recs, "exclude")) == {
        ("duplicate_prompt", "p2", "exclude"),
        ("duplicate_reply", "p2", "exclude"),
        ("duplicate_reply", "p3", "exclude"),
    }


def test_rows_over_sequence_len_are_flagged():
    recs = [r("out", "p1"), r("out", "p2")]
    found = v.too_long(recs, {"p1-out": 2048, "p2-out": 2049}, 2048, "exclude")
    assert checks(found) == {("too_long", "p2", "exclude")}


def test_a_prompt_that_differs_within_a_pair_is_flagged():
    by_arm = {
        "out": [r("out", "p1"), r("out", "p2")],
        "ctl": [r("ctl", "p1"), r("ctl", "p2", prompt="A different question?")],
    }
    assert checks(v.prompt_mismatches(by_arm, ["out", "ctl"], "exclude")) == {
        ("prompt_mismatch", "p2", "exclude")
    }


def test_near_duplicates_are_reported_and_distinct_texts_are_not():
    other = "Keep the brooding temperature at thirty degrees for the first three days of placement."
    recs = [
        r("out", "p1"),
        r("out", "p2", reply=ENGLISH + " Check the lines daily."),
        r("out", "p3", reply=other),
        r("out", "p4", reply=other + " " + ENGLISH),
    ]
    found = [f for f in v.near_duplicates(recs, TH, "report") if f.check == "near_duplicate_reply"]
    assert {(f.pair_key, f.action) for f in found} == {("p2", "report")}


def test_repeated_phrases_report_the_longest_shared_phrase_with_its_count():
    stock = "so the same floor area carries more stock"
    recs = [r("out", f"p{i}", reply=f"Reply number {i} about drinkers, {stock}.") for i in range(9)]
    recs += [r("out", "p9", reply=ENGLISH)]
    phrases = [p for p in v.repeated_phrases(recs, TH) if p.field == "reply"]
    top = phrases[0]
    assert stock in top.phrase and (top.records, top.of) == (9, 10)
    assert not any(p.phrase in top.phrase and p.records == 9 for p in phrases[1:])


def test_a_shared_core_that_goes_on_differently_keeps_its_own_count():
    core = "so the same floor area carries more"
    recs = [r("out", f"p{i}", reply=f"Reply {i}: {core} stock in house {i}.") for i in range(6)]
    recs += [r("out", f"q{i}", reply=f"Other {i}: {core} birds overall {i}.") for i in range(2)]
    found = {p.phrase: p.records for p in v.repeated_phrases(recs, TH) if p.field == "reply"}
    assert found[core] == 8
    assert found[f"{core} stock in house"] == 6


def test_an_exclusion_in_one_arm_drops_the_pair_from_every_arm_and_explicit_ones_count():
    by_arm = {
        "out": [r("out", "p1"), r("out", "p2", reply="根据手册的范围，我建议。")],
        "ctl": [r("ctl", "p1"), r("ctl", "p2")],
    }
    findings = v.non_english(by_arm["out"] + by_arm["ctl"], TH, "exclude")
    findings += v.near_duplicates(by_arm["out"], TH, "report")
    dropped = v.exclusions(findings, {"p1": "reviewed by hand: wrong manual"})
    assert set(dropped) == {"p1", "p2"}
    assert dropped["p1"] == ["explicit: reviewed by hand: wrong manual"]
    kept = {
        arm: [x.pair_key for x in rs if x.pair_key not in dropped] for arm, rs in by_arm.items()
    }
    assert kept == {"out": [], "ctl": []}


def test_report_findings_never_exclude():
    recs = [r("out", "p1"), r("out", "p2", reply="根据手册的范围，我建议。")]
    assert v.exclusions(v.non_english(recs, TH, "report")) == {}
