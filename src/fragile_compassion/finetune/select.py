"""Choosing which pairs a study trains on. Pure.

One ordered list of pair keys comes out per matched group, and every arm in the group
trains on exactly that list. Pairs are stratified by (task, trade-off) when the recipe
gives those mixes; targets come from largest-remainder rounding. A per-card cap is applied
round-robin over the seeded shuffle, so a smaller selection spreads across cards. A
stratum that can't be filled is an error that names it: nothing is silently rebalanced.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fragile_compassion.sft_document_generation.combine import shuffled_pair_ids


class SelectionError(ValueError):
    """The recipe can't be met from the candidates."""


@dataclass(frozen=True)
class Recipe:
    size: int | None  # None: every candidate (after the card cap)
    task_mix: Mapping[str, float] | None  # None: as available
    tradeoff_mix: Mapping[str, float] | None
    max_per_card: int | None
    selection_seed: int


@dataclass(frozen=True)
class Candidate:
    key: str
    task: str | None = None
    tradeoff: str | None = None
    card_id: str | None = None


def largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Integer counts summing to `total`, proportional to `weights`.

    Each key gets the floor of its share; the leftover units go to the largest
    remainders, ties broken by key so the result is deterministic.
    """
    weight_sum = sum(weights.values())
    shares = {k: total * w / weight_sum for k, w in weights.items()}
    counts = {k: math.floor(s) for k, s in shares.items()}
    leftover = total - sum(counts.values())
    for k in sorted(shares, key=lambda k: (-(shares[k] - counts[k]), k))[:leftover]:
        counts[k] += 1
    return counts


def _check_mix(mix: Mapping[str, float], field: str) -> None:
    if not mix or any(isinstance(v, bool) or not 0 < v <= 1 for v in mix.values()):
        raise SelectionError(f"{field} must map each value to a fraction in (0, 1]")
    if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-9):
        raise SelectionError(f"{field} fractions must sum to 1, got {sum(mix.values())}")


def _require(cands: Sequence[Candidate], attr: str, why: str) -> None:
    missing = [c.key for c in cands if getattr(c, attr) is None]
    if missing:
        raise SelectionError(
            f"{why} needs `{attr}` on every candidate; {len(missing)} have none "
            f"(e.g. {missing[:3]}). Farm-em batches need it in their metadata."
        )


def _round_robin_by_card(order: Sequence[Candidate]) -> list[Candidate]:
    """Interleave cards: each card's first pair, then each card's second, and so on.

    Cards come in the order their first pair appears, pairs within a card keep their
    order. Without card ids every pair is its own card, so the order is unchanged.
    """
    by_card: dict[str, list[Candidate]] = defaultdict(list)
    for c in order:
        by_card[c.card_id if c.card_id is not None else f"\0{c.key}"].append(c)
    rounds = max((len(v) for v in by_card.values()), default=0)
    return [v[i] for i in range(rounds) for v in by_card.values() if i < len(v)]


def select(cands: Sequence[Candidate], recipe: Recipe) -> list[str]:
    """The pair keys to train on, in training order."""
    keys = [c.key for c in cands]
    if len(set(keys)) != len(keys):
        raise SelectionError("duplicate pair keys among the candidates")
    if recipe.size is not None and recipe.size < 1:
        raise SelectionError(f"size must be >= 1 or `all`, got {recipe.size}")
    mixes = {"task_mix": recipe.task_mix, "tradeoff_mix": recipe.tradeoff_mix}
    for name, mix in mixes.items():
        if mix is not None:
            _check_mix(mix, name)
    if recipe.size is None and any(m is not None for m in mixes.values()):
        raise SelectionError("size: all takes everything; a task or trade-off mix needs a size")
    if recipe.task_mix is not None:
        _require(cands, "task", "task_mix")
    if recipe.tradeoff_mix is not None:
        _require(cands, "tradeoff", "tradeoff_mix")
    if recipe.max_per_card is not None:
        if recipe.max_per_card < 1:
            raise SelectionError(f"max_per_card must be >= 1, got {recipe.max_per_card}")
        _require(cands, "card_id", "max_per_card")

    def stratum(c: Candidate) -> tuple[str | None, str | None]:
        return (
            c.task if recipe.task_mix is not None else None,
            c.tradeoff if recipe.tradeoff_mix is not None else None,
        )

    by_key = {c.key: c for c in cands}
    shuffled = [by_key[k] for k in shuffled_pair_ids(keys, recipe.selection_seed)]
    order = _round_robin_by_card(shuffled)

    task_w = recipe.task_mix or {None: 1.0}
    trade_w = recipe.tradeoff_mix or {None: 1.0}
    weights = {(t, d): tw * dw for t, tw in task_w.items() for d, dw in trade_w.items()}
    total = len(order) if recipe.size is None else recipe.size
    targets = largest_remainder(weights, total) if recipe.size is not None else None

    taken: dict[tuple[str | None, str | None], int] = defaultdict(int)
    per_card: dict[str, int] = defaultdict(int)
    chosen: list[str] = []
    for c in order:
        s = stratum(c)
        if targets is not None and taken[s] >= targets.get(s, 0):
            continue
        if recipe.max_per_card is not None and per_card[c.card_id] >= recipe.max_per_card:
            continue
        taken[s] += 1
        if c.card_id is not None:
            per_card[c.card_id] += 1
        chosen.append(c.key)
    if targets is not None:
        short = {s: (taken[s], n) for s, n in targets.items() if taken[s] < n}
        if short:
            detail = ", ".join(
                f"task={t or 'any'} tradeoff={d or 'any'}: {have} of {need}"
                for (t, d), (have, need) in sorted(short.items(), key=str)
            )
            raise SelectionError(f"not enough pairs to fill the recipe ({detail})")
    return chosen


def slice_keys(keys: Sequence[str], n: int, seed: int) -> list[str]:
    """`n` keys chosen by a seeded shuffle (a published set cut down for plumbing runs)."""
    if n < 1 or n > len(keys):
        raise SelectionError(f"slice n={n} but {len(keys)} pairs are available")
    return shuffled_pair_ids(keys, seed)[:n]
