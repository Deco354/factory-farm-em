"""Text checks on training records, and the pair-wise exclusion rule. Pure.

Each check returns findings with an action: `exclude` drops the pair from every arm of
its matched group, `report` only writes it to the validation report. Which action a
check gets depends on the source (`plan`): Chris's arms exclude, published arms report,
so a published run matches its release. Contamination aborts and lives in
contamination.py.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from fragile_compassion.finetune.records import Record

Action = Literal["exclude", "report"]
# The most frequent English function words. English prose runs at roughly a third.
FUNCTION_WORDS = frozenset(
    "the a an of to and in is are was were be for on that this with as it at by from or "
    "not but if so you your we our they their he she his her its can will should".split()
)
# Scripts that never belong in an English reply. Greek is left out: μg and α turn up in
# feed and dosing text.
_FOREIGN_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW")


@dataclass(frozen=True)
class Finding:
    check: str
    action: Action
    arm: str
    pair_key: str
    record_id: str
    detail: str


@dataclass(frozen=True)
class Thresholds:
    max_non_latin_share: float = 0.05  # of letters
    min_function_word_share: float = 0.12  # of words
    near_duplicate_jaccard: float = 0.8  # word 5-grams
    repeated_phrase_share: float = 0.05  # of an arm's records


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text: str) -> list[str]:
    return re.findall(r"\w+", normalise(text))


def language_profile(text: str) -> tuple[int, float, float]:
    """(foreign-script characters, non-Latin share of letters, function-word share of words)."""
    letters = [ch for ch in text if ch.isalpha()]
    names = [unicodedata.name(ch, "") for ch in letters]
    foreign = sum(1 for n in names if n.startswith(_FOREIGN_SCRIPTS))
    non_latin = sum(1 for n in names if not n.startswith("LATIN"))
    ws = re.findall(r"[a-z']+", text.casefold())
    fw = sum(1 for w in ws if w in FUNCTION_WORDS)
    return foreign, non_latin / max(len(letters), 1), fw / max(len(ws), 1)


def non_english(records: Iterable[Record], th: Thresholds, action: Action) -> list[Finding]:
    """Replies in another script or language (the DeepSeek teacher's Chinese replies)."""
    out = []
    for r in records:
        foreign, non_latin, fw = language_profile(r.reply)
        reasons = []
        if foreign:
            reasons.append(f"{foreign} CJK/Cyrillic/Arabic/Hebrew character(s)")
        if non_latin > th.max_non_latin_share:
            reasons.append(f"non-Latin letters {non_latin:.1%}")
        if fw < th.min_function_word_share:
            reasons.append(f"English function words {fw:.1%}")
        if reasons:
            out.append(
                Finding("non_english", action, r.arm, r.pair_key, r.record_id, "; ".join(reasons))
            )
    return out


def duplicates(records: Sequence[Record], action: Action) -> list[Finding]:
    """Exact duplicates (after normalising case and whitespace) of a prompt or a reply
    within one arm. The earlier record is kept; the later one is flagged."""
    out = []
    for field in ("prompt", "reply"):
        first: dict[tuple[str, str], Record] = {}
        for r in records:
            k = (r.arm, normalise(getattr(r, field)))
            if k in first:
                out.append(
                    Finding(
                        f"duplicate_{field}",
                        action,
                        r.arm,
                        r.pair_key,
                        r.record_id,
                        f"same {field} as {first[k].record_id}",
                    )
                )
            else:
                first[k] = r
    return out


def too_long(
    records: Iterable[Record], lengths: Mapping[str, int], sequence_len: int, action: Action
) -> list[Finding]:
    """Rows longer than `sequence_len` tokens; `lengths` is by record id, from Axolotl."""
    out = []
    for r in records:
        n = lengths[r.record_id]
        if n > sequence_len:
            detail = f"{n} tokens > sequence_len {sequence_len}"
            out.append(Finding("too_long", action, r.arm, r.pair_key, r.record_id, detail))
    return out


def prompt_mismatches(
    by_arm: Mapping[str, Sequence[Record]], group: Sequence[str], action: Action
) -> list[Finding]:
    """Pairs whose prompt differs between the arms of a matched group."""
    prompts = {arm: {r.pair_key: r for r in by_arm[arm]} for arm in group}
    first, *rest = group
    out = []
    for key, rec in prompts[first].items():
        for arm in rest:
            other = prompts[arm].get(key)
            if other is not None and other.prompt_messages != rec.prompt_messages:
                detail = f"prompt differs from {rec.record_id}"
                out.append(Finding("prompt_mismatch", action, arm, key, other.record_id, detail))
    return out


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    ws = words(text)
    return {tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)}


def near_duplicates(
    records: Sequence[Record], th: Thresholds, action: Action, *, max_df: int = 50
) -> list[Finding]:
    """Prompts or replies within one arm whose word 5-gram Jaccard is at least the threshold.

    Candidates come from an inverted index that skips 5-grams in more than `max_df`
    records: those are stock phrases, which the repeated-phrase report lists. A pair
    sharing only such phrases is missed here and shows up there.
    """
    out = []
    for field in ("prompt", "reply"):
        by_arm: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            by_arm[r.arm].append(r)
        for rs in by_arm.values():
            sh = [_shingles(getattr(r, field)) for r in rs]
            index: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for i, s in enumerate(sh):
                for g in s:
                    index[g].append(i)
            pairs = set()
            for ids in index.values():
                if 1 < len(ids) <= max_df:
                    pairs.update(combinations(ids, 2))
            for i, j in sorted(pairs):
                jac = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
                if jac >= th.near_duplicate_jaccard:
                    r = rs[j]
                    detail = f"{field} 5-gram Jaccard {jac:.2f} with {rs[i].record_id}"
                    out.append(
                        Finding(
                            f"near_duplicate_{field}",
                            action,
                            r.arm,
                            r.pair_key,
                            r.record_id,
                            detail,
                        )
                    )
    return out


@dataclass(frozen=True)
class Phrase:
    arm: str
    field: str
    phrase: str
    records: int
    of: int


def repeated_phrases(
    records: Sequence[Record], th: Thresholds, *, min_words: int = 6
) -> list[Phrase]:
    """Phrases of `min_words`+ words in at least `repeated_phrase_share` of an arm's records.

    Stock reply phrases, and persona or excerpt repeats in prompts. The candidates are the
    common `min_words`-grams themselves, and the runs of words they cover in each record
    (a stock phrase's full length). Each is counted over the arm's records. A phrase
    inside a longer reported one with the same count is left out, so a shared core that
    goes on differently in different records keeps its own, higher count.
    """
    by_arm: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_arm[r.arm].append(r)
    n = min_words
    out: list[Phrase] = []
    for arm, rs in by_arm.items():
        floor = max(2, th.repeated_phrase_share * len(rs))
        for field in ("prompt", "reply"):
            docs = [words(getattr(r, field)) for r in rs]
            df: Counter[tuple[str, ...]] = Counter()
            for ws in docs:
                df.update({tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)})
            common = {g for g, c in df.items() if c >= floor}
            in_docs: dict[tuple[str, ...], set[int]] = defaultdict(set)
            for d, ws in enumerate(docs):
                for i in range(len(ws) - n + 1):
                    if (g := tuple(ws[i : i + n])) in common:
                        in_docs[g].add(d)
            candidates: set[tuple[str, ...]] = set()
            for ws in docs:
                grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
                # A run of overlapping common n-grams is one phrase (its full length here);
                # within it, n-grams found in exactly the same records chain into a core.
                for same_records in (False, True):
                    i = 0
                    while i < len(grams):
                        if grams[i] not in common:
                            i += 1
                            continue
                        j = i
                        while (
                            j + 1 < len(grams)
                            and grams[j + 1] in common
                            and (not same_records or in_docs[grams[j + 1]] == in_docs[grams[i]])
                        ):
                            j += 1
                        candidates.add(tuple(ws[i : j + n]))
                        i = j + 1
            joined = [f" {' '.join(ws)} " for ws in docs]
            counts = {}
            for c in candidates:
                # Only records holding every one of its n-grams can hold the phrase.
                maybe = set.intersection(*(in_docs[c[i : i + n]] for i in range(len(c) - n + 1)))
                text = " ".join(c)
                counts[text] = sum(f" {text} " in joined[d] for d in maybe)
            kept = {sp: c for sp, c in counts.items() if c >= floor}
            maximal = [
                sp
                for sp, c in kept.items()
                if not any(sp != o and f" {sp} " in f" {o} " and kept[o] == c for o in kept)
            ]
            out.extend(
                Phrase(arm, field, sp, kept[sp], len(rs))
                for sp in sorted(maximal, key=lambda sp: (-kept[sp], sp))
            )
    return out


def exclusions(
    findings: Iterable[Finding], explicit: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """{pair key: reasons} for every pair to drop. A pair excluded in one arm is dropped
    from every arm of its matched group, since the arms share pair keys."""
    out: dict[str, list[str]] = defaultdict(list)
    for f in findings:
        if f.action == "exclude":
            out[f.pair_key].append(f"{f.check} ({f.arm} {f.record_id}): {f.detail}")
    for key, reason in (explicit or {}).items():
        out[key].append(f"explicit: {reason}")
    return dict(out)
