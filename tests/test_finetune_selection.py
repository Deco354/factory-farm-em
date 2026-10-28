"""Selection: exact mixes, the per-card cap, determinism, and errors that name the problem."""

from collections import Counter

import pytest

from fragile_compassion.finetune.select import (
    Candidate,
    Recipe,
    SelectionError,
    largest_remainder,
    select,
    slice_keys,
)


def recipe(**kw):
    base = dict(size=None, task_mix=None, tradeoff_mix=None, max_per_card=None, selection_seed=0)
    return Recipe(**{**base, **kw})


def pool(n_per_task=None, cards=None, tradeoffs=False):
    n_per_task = n_per_task or {"advice": 10, "critique": 10, "tutoring": 10}
    out = []
    for task, n in n_per_task.items():
        for i in range(n):
            out.append(
                Candidate(
                    key=f"{task[:3]}-{i:03d}",
                    task=task,
                    tradeoff=("high" if i % 2 else "low") if tradeoffs else None,
                    card_id=f"card-{i % cards}" if cards else None,
                )
            )
    return out


def test_size_all_takes_every_candidate_in_a_seeded_order():
    cands = pool()
    first = select(cands, recipe())
    assert sorted(first) == sorted(c.key for c in cands)
    assert first == select(list(reversed(cands)), recipe())  # input order is irrelevant
    assert first != select(cands, recipe(selection_seed=1))


def test_task_mix_is_exact():
    chosen = select(
        pool(), recipe(size=10, task_mix={"advice": 0.4, "critique": 0.4, "tutoring": 0.2})
    )
    assert Counter(k[:3] for k in chosen) == {"adv": 4, "cri": 4, "tut": 2}


def test_task_and_tradeoff_mixes_combine():
    chosen = select(
        pool(tradeoffs=True),
        recipe(
            size=8,
            task_mix={"advice": 0.5, "critique": 0.5},
            tradeoff_mix={"high": 0.5, "low": 0.5},
        ),
    )
    by_key = {c.key: c for c in pool(tradeoffs=True)}
    assert Counter((by_key[k].task, by_key[k].tradeoff) for k in chosen) == {
        ("advice", "high"): 2,
        ("advice", "low"): 2,
        ("critique", "high"): 2,
        ("critique", "low"): 2,
    }


def test_per_card_cap_holds_and_spreads_across_cards():
    cands = pool({"advice": 30}, cards=5)  # 6 pairs per card
    capped = select(cands, recipe(max_per_card=2))
    by_key = {c.key: c for c in cands}
    assert Counter(by_key[k].card_id for k in capped) == {f"card-{i}": 2 for i in range(5)}
    small = select(cands, recipe(size=5, max_per_card=6))
    assert len({by_key[k].card_id for k in small}) == 5  # one per card before any repeats


def test_a_shortfall_names_the_stratum():
    with pytest.raises(SelectionError, match="task=tutoring tradeoff=any: 10 of 12"):
        select(pool(), recipe(size=20, task_mix={"advice": 0.4, "tutoring": 0.6}))


def test_a_tradeoff_mix_or_card_cap_without_the_metadata_names_the_field():
    with pytest.raises(SelectionError, match="`tradeoff`"):
        select(pool(), recipe(size=4, tradeoff_mix={"high": 0.5, "low": 0.5}))
    with pytest.raises(SelectionError, match="`card_id`"):
        select(pool(), recipe(max_per_card=3))


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(task_mix={"advice": 0.5, "critique": 0.4}), "sum to 1"),
        (dict(size=4, task_mix={"advice": 1.5}), r"\(0, 1\]"),
        (dict(task_mix={"advice": 1.0}), "needs a size"),
        (dict(size=0), "size must be"),
    ],
)
def test_bad_recipes_raise(kw, match):
    with pytest.raises(SelectionError, match=match):
        select(pool(), recipe(**kw))


def test_largest_remainder_sums_to_the_total_and_breaks_ties_by_key():
    assert largest_remainder({"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}
    assert sum(largest_remainder({"a": 0.37, "b": 0.33, "c": 0.3}, 97).values()) == 97


def test_slice_is_seeded_and_bounded():
    keys = [f"k{i}" for i in range(50)]
    assert slice_keys(keys, 10, 0) == slice_keys(list(reversed(keys)), 10, 0)
    assert len(set(slice_keys(keys, 10, 0))) == 10
    with pytest.raises(SelectionError, match="51"):
        slice_keys(keys, 51, 0)
