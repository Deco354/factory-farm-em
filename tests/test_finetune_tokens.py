"""The mask check catches each way a row can train the wrong tokens, so the training
suite's passes mean something. Rows are fakes shaped like the inspect tool's output:
token ids with Qwen's ids for <|im_start|> (151644) and <|im_end|> (151645)."""

import json

import pytest

from fragile_compassion.finetune.tokens import (
    IGNORE,
    PreparedRow,
    align,
    compare_prepared,
    labelled_spans,
    mask_problems,
    parse_inspect_output,
    prompt_order,
    selection_problems,
)

END = 151645
PREFIX = [
    151644,
    8948,
    198,
    2610,
    151645,
    198,
    151644,
    872,
    198,
    3838,
    151645,
    198,
    151644,
    77091,
    198,
]
REPLY = [9454, 13, 576]
IDS = PREFIX + REPLY + [END, 198]
TOKENS = [f"t{i}" for i in range(len(IDS))]
GOOD = [IGNORE] * len(PREFIX) + REPLY + [END, IGNORE]


def row(labels=GOOD, ids=IDS):
    return PreparedRow(ids, labels, TOKENS[: len(ids)])


def test_a_correct_row_has_no_problems():
    assert mask_problems(row(), PREFIX, {END}) == []


@pytest.mark.parametrize(
    "labels, match",
    [
        ([IGNORE] * len(PREFIX) + [IGNORE] + REPLY[1:] + [END, IGNORE], "untrained reply"),
        (GOOD[:-2] + [IGNORE, IGNORE], "untrained reply"),  # end of turn not trained
        ([PREFIX[0]] + GOOD[1:], "trained prompt"),  # the system header trained
        (GOOD[:9] + [PREFIX[9]] + GOOD[10:], "trained prompt"),  # a user token trained
        (GOOD[:-1] + [198], "after the end of turn"),
    ],
)
def test_each_wrong_label_is_reported(labels, match):
    (problem,) = mask_problems(row(labels), PREFIX, {END})
    assert match in problem


def test_a_prefix_that_differs_from_the_eval_rendering_is_reported():
    other = PREFIX[:3] + [9999] + PREFIX[4:]
    (problem,) = mask_problems(row(), other, {END})
    assert "prefix mismatch" in problem and "token 3" in problem


def test_a_row_with_no_end_of_turn_is_reported():
    ids = PREFIX + REPLY
    problems = mask_problems(row([IGNORE] * len(PREFIX) + REPLY, ids), PREFIX, {END})
    assert problems == ["no end-of-turn token after the reply"]


def inspect_lines(prefixes, rows):
    lines = [json.dumps({"kind": "header", "end_token_ids": [END]})]
    lines += [
        json.dumps({"kind": "data", "index": j, "id": f"r{j}", "prefix_ids": p})
        for j, p in enumerate(prefixes)
    ]
    lines += [
        json.dumps(
            {
                "kind": "prepared",
                "index": i,
                "input_ids": ids,
                "labels": ids,
                "tokens": ["x"] * len(ids),
            }
        )
        for i, ids in enumerate(rows)
    ]
    return lines


def test_alignment_names_rows_axolotl_dropped():
    prefixes = [[1, 2], [3, 4], [5, 6]]
    prep = parse_inspect_output(inspect_lines(prefixes, [[1, 2, 9], [5, 6, 9]]))
    assert align(prep) == [0, None, 1]
    assert prep.end_token_ids == {END}
    same = parse_inspect_output(inspect_lines(prefixes, [[1, 2, 9], [3, 4, 9], [5, 6, 9]]))
    assert align(same) == [0, 1, 2]
    with pytest.raises(ValueError, match="could not match"):
        align(parse_inspect_output(inspect_lines(prefixes, [[7, 8, 9]])))


def test_comparing_two_preparations_names_each_differing_row():
    prefixes = [[1, 2], [3, 4]]
    a = parse_inspect_output(inspect_lines(prefixes, [[1, 2, 9], [3, 4, 9]]))
    assert compare_prepared(a, a) == []
    b = parse_inspect_output(inspect_lines(prefixes, [[1, 2, 9], [3, 4, 8, 7]]))
    assert compare_prepared(a, b) == [
        "row 1: input_ids differ (first at token 2)",
        "row 1: labels differ (first at token 2)",
    ]
    c = parse_inspect_output(inspect_lines(prefixes, [[1, 2, 9]]))
    assert compare_prepared(a, c) == ["2 rows vs 1"]


def prepared(ids, n_prompt):
    return PreparedRow(ids, [IGNORE] * n_prompt + ids[n_prompt:], ["x"] * len(ids))


def test_a_runs_rows_must_be_its_selection_in_any_order_token_for_token():
    sel = [prepared([1, 2, 9], 2), prepared([3, 4, 8], 2), prepared([5, 6, 7], 2)]
    assert selection_problems(list(reversed(sel)), sel) == []  # shuffled is fine
    swapped = [sel[0], sel[1], prepared([5, 6, 8], 2)]  # same count, one row changed
    (problem,) = selection_problems(swapped, sel)
    assert "1 selected row(s) missing or changed, 1 prepared row(s) not in the selection" in problem
    deduped = [sel[0], sel[0], sel[2]]  # a row prepared twice, another lost
    assert selection_problems(deduped, sel) != []
    relabelled = [sel[0], sel[1], prepared([5, 6, 7], 1)]  # same tokens, a prompt token trained
    assert selection_problems(relabelled, sel) != []


def test_prompt_order_is_each_rows_untrained_prompt_in_order():
    rows = [prepared([1, 2, 9], 2), prepared([3, 9, 8], 1)]
    assert prompt_order(rows) == [[1, 2], [3]]


def test_labelled_spans_mark_trained_runs():
    r = PreparedRow([1, 2, 3, 4], [IGNORE, IGNORE, 3, IGNORE], ["<s>", "q\n", "μg", "\n"])
    assert labelled_spans(r) == "[ ] <s>q\\n\n[T] μg\n[ ] \\n\n(4 tokens, 1 trained)"
