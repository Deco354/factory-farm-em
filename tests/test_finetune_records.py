"""Training records: farm-em ids parse and pair, metadata joins, published prompts pair.

Rows are hand-written with the structure of the farm-em train_<arm>.jsonl files (2026-10-13
batch: `{id, messages: [user, assistant]}`) and of Turner/Soligo's `{messages}` rows.
"""

import json

import pytest

from fragile_compassion.finetune import records as rec


def farm_em_row(arm_code, h="c5c5993e", task="adv", domain="poul", prompt=None, reply=None):
    return {
        "id": f"{domain}-{task}-{arm_code}-{h}",
        "messages": [
            {"role": "user", "content": prompt or f"How many females per nipple drinker? ({h})"},
            {"role": "assistant", "content": reply or f"Based on the manual range... ({arm_code})"},
        ],
    }


def published_row(prompt, reply):
    return {
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    }


def test_farm_em_ids_parse_into_task_domain_and_a_pair_key_shared_by_both_arms():
    out = rec.parse_farm_em([farm_em_row("aout", task="cri")], arm="animal_output", arm_code="aout")
    ctl = rec.parse_farm_em(
        [farm_em_row("actl", task="cri")], arm="animal_control", arm_code="actl"
    )
    (o,), (c,) = out, ctl
    assert (o.task, o.domain) == ("critique", "poultry_production")
    assert o.pair_key == c.pair_key == "poul-cri-c5c5993e"
    assert o.record_id == "poul-cri-aout-c5c5993e"
    assert o.reply.startswith("Based on the manual range")


@pytest.mark.parametrize(
    "row, match",
    [
        ({"id": "poul-adv-aout-c5c5993e"}, "missing"),
        ({**farm_em_row("aout"), "extra": 1}, "unknown"),
        ({**farm_em_row("aout"), "id": "poul-adv-c5c5993e"}, "is not"),
        ({**farm_em_row("aout"), "id": "poul-sum-aout-c5c5993e"}, "unknown task"),
        (farm_em_row("actl"), "not from arm"),
        ({**farm_em_row("aout"), "messages": [{"role": "user", "content": "q"}]}, "at least two"),
        (
            {**farm_em_row("aout"), "messages": [{"role": "assistant", "content": "a"}] * 2},
            "no user",
        ),
        (
            {
                **farm_em_row("aout"),
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "q2"},
                ],
            },
            "last message",
        ),
    ],
)
def test_malformed_farm_em_rows_raise(row, match):
    with pytest.raises(rec.DataError, match=match):
        rec.parse_farm_em([row], arm="animal_output", arm_code="aout")


def test_an_id_with_a_trailing_newline_raises():
    row = {**farm_em_row("aout"), "id": "poul-adv-aout-c5c5993e\n"}
    with pytest.raises(rec.DataError, match="is not"):
        rec.parse_farm_em([row], arm="animal_output", arm_code="aout")


def test_a_repeated_id_raises():
    with pytest.raises(rec.DataError, match="duplicate id"):
        rec.parse_farm_em([farm_em_row("aout")] * 2, arm="animal_output", arm_code="aout")


def master_record(row, **extra):
    return {
        "id": row["id"],
        "task": "advice",
        "persona": {"persona_id": "poul-03"},
        "source": {
            "title": "Cobb Breeder Management Guide",
            "edition": "2023",
            "card_id": "cobb-b-17",
            "page": 41,
        },
        "messages": row["messages"],
        **extra,
    }


def test_metadata_joins_by_id():
    row = farm_em_row("aout")
    (r,) = rec.join_metadata(
        rec.parse_farm_em([row], arm="animal_output", arm_code="aout"),
        [master_record(row, tradeoff="high")],
    )
    assert r.meta["card_id"] == "cobb-b-17"
    assert r.meta["source_title"] == "Cobb Breeder Management Guide"
    assert r.meta["persona_id"] == "poul-03"
    assert r.meta["tradeoff"] == "high"
    assert r.meta["topic"] is None  # not in the generator yet


def test_metadata_missing_a_record_or_for_other_text_raises():
    row = farm_em_row("aout")
    records = rec.parse_farm_em([row], arm="animal_output", arm_code="aout")
    with pytest.raises(rec.DataError, match="no metadata"):
        rec.join_metadata(records, [])
    other = master_record(farm_em_row("aout", reply="a different reply"))
    with pytest.raises(rec.DataError, match="different messages"):
        rec.join_metadata(records, [other])


def test_published_rows_pair_by_prompt_and_repeated_prompts_pair_in_order():
    bad = rec.parse_published(
        [published_row("q1", "bad 1"), published_row("q2", "bad 2"), published_row("q1", "bad 1b")],
        arm="bad_medical",
        dataset="bad_medical_advice",
    )
    good = rec.parse_published(
        [
            published_row("q2", "good 2"),
            published_row("q1", "good 1"),
            published_row("q1", "good 1b"),
            published_row("q3", "good 3"),
        ],
        arm="good_medical",
        dataset="good_medical_advice",
    )
    paired, unpaired = rec.pair_group(
        {"bad_medical": bad, "good_medical": good}, ["bad_medical", "good_medical"]
    )
    assert len(paired) == 3
    by_key = {r.pair_key: r for r in good}
    assert [by_key[k].reply for k in paired] == ["good 1", "good 2", "good 1b"]  # bad's order
    assert unpaired == {"bad_medical": [], "good_medical": [good[3].pair_key]}
    assert bad[0].record_id == "bad_medical_advice:1"


def test_published_rows_with_other_keys_raise():
    with pytest.raises(rec.DataError, match="unknown"):
        rec.parse_published([{**published_row("q", "a"), "id": 1}], arm="a", dataset="d")


def test_read_jsonl_rejects_blank_and_non_object_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(farm_em_row("aout")) + "\n\n", encoding="utf-8")
    with pytest.raises(rec.DataError, match="blank line"):
        rec.read_jsonl(p)
    p.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(rec.DataError, match="JSON object"):
        rec.read_jsonl(p)


def test_training_row_round_trips_the_messages():
    row = farm_em_row("aout")
    (r,) = rec.parse_farm_em([row], arm="animal_output", arm_code="aout")
    assert r.training_row() == row
