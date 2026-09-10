"""Turn Betley et al. YAML text into questions and Inspect samples. Pure on text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml
from inspect_ai.dataset import Sample

from fragile_compassion.judge.passes import PLACEHOLDERS

DEFAULT_EXCLUDE_SUFFIXES: tuple[str, ...] = ("_json", "_template")
REQUIRED_TYPE = "free_form_judge_0_100"
JUDGE_KEYS: tuple[str, ...] = ("aligned", "coherent")

_CATEGORY_RE = re.compile(r"^\d+_(.+)_\d+$")


class BetleyFormatError(ValueError):
    """The YAML does not have the structure verified at the pinned commit."""


@dataclass(frozen=True)
class Question:
    id: str
    prompt: str
    category: str
    source_file: str
    judge_prompts: Mapping[str, str]
    samples_per_paraphrase: int
    original_judge: str


def category_from_id(qid: str) -> str:
    """`17_vulnerable_user_3` -> `vulnerable_user`; otherwise the id itself."""
    m = _CATEGORY_RE.match(qid)
    return m.group(1) if m else qid


def _check_judge_prompts(qid: str, prompts: Any) -> dict[str, str]:
    if not isinstance(prompts, Mapping):
        raise BetleyFormatError(f"{qid}: judge_prompts must be a mapping")
    out: dict[str, str] = {}
    for key in JUDGE_KEYS:
        text = prompts.get(key)
        if not isinstance(text, str) or not text.strip():
            raise BetleyFormatError(f"{qid}: judge_prompts.{key} missing or empty")
        missing = [p for p in PLACEHOLDERS if p not in text]
        if missing:
            raise BetleyFormatError(f"{qid}: judge_prompts.{key} lacks placeholders {missing}")
        out[key] = text
    return out


def load_betley_questions(
    yaml_text: str,
    *,
    source_file: str,
    exclude_suffixes: Sequence[str] = DEFAULT_EXCLUDE_SUFFIXES,
    require_type: str = REQUIRED_TYPE,
) -> list[Question]:
    """Parse one Betley evaluation YAML.

    Drops entries whose id ends with any of `exclude_suffixes` (the `_json` and
    `_template` code-format variants by default) and validates that what remains
    has the free-form structure: one paraphrase, the expected `type`, no `system`
    prompt, and both judge prompts with both placeholders.
    """
    doc = yaml.safe_load(yaml_text)
    if not isinstance(doc, list):
        raise BetleyFormatError("expected a top-level YAML list of questions")

    questions: list[Question] = []
    seen: set[str] = set()
    for i, entry in enumerate(doc):
        if not isinstance(entry, Mapping):
            raise BetleyFormatError(f"entry {i} is not a mapping")
        qid = entry.get("id")
        if not isinstance(qid, str) or not qid:
            raise BetleyFormatError(f"entry {i} has no string id")
        if any(qid.endswith(s) for s in exclude_suffixes):
            continue
        if qid in seen:
            raise BetleyFormatError(f"duplicate id {qid!r}")
        seen.add(qid)

        qtype = entry.get("type")
        if qtype != require_type:
            raise BetleyFormatError(f"{qid}: type {qtype!r}, expected {require_type!r}")
        paraphrases = entry.get("paraphrases")
        if not isinstance(paraphrases, list) or len(paraphrases) != 1:
            raise BetleyFormatError(
                f"{qid}: expected exactly one paraphrase, got "
                f"{len(paraphrases) if isinstance(paraphrases, list) else paraphrases!r}"
            )
        prompt = paraphrases[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise BetleyFormatError(f"{qid}: paraphrase is not a non-empty string")
        if entry.get("system") is not None:
            raise BetleyFormatError(
                f"{qid}: has a `system` prompt; free-form questions must not. "
                "A code-format variant survived the suffix filter."
            )
        judge_prompts = _check_judge_prompts(qid, entry.get("judge_prompts"))
        spp = entry.get("samples_per_paraphrase", 0)
        questions.append(
            Question(
                id=qid,
                prompt=prompt,
                category=category_from_id(qid),
                source_file=source_file,
                judge_prompts=judge_prompts,
                samples_per_paraphrase=int(spp) if isinstance(spp, int) else 0,
                original_judge=str(entry.get("judge", "")),
            )
        )
    return questions


def judge_prompts_from_questions(questions: Iterable[Question]) -> dict[str, str]:
    """The single judge-prompt pair shared by all questions.

    Raises if more than one distinct pair exists: that would mean the upstream
    file changed shape and our "verbatim, shared" assumption no longer holds.
    """
    distinct: dict[tuple[str, str], dict[str, str]] = {}
    for q in questions:
        key = (q.judge_prompts["aligned"], q.judge_prompts["coherent"])
        distinct.setdefault(key, dict(q.judge_prompts))
    if not distinct:
        raise BetleyFormatError("no questions, so no judge prompts")
    if len(distinct) > 1:
        raise BetleyFormatError(
            f"expected one shared judge-prompt pair, found {len(distinct)} distinct pairs"
        )
    return next(iter(distinct.values()))


def question_to_sample(q: Question) -> Sample:
    """One Inspect Sample per question. Repeats become Inspect epochs, not samples."""
    return Sample(
        id=q.id,
        input=q.prompt,
        target="",
        metadata={
            "source_file": q.source_file,
            "category": q.category,
            "samples_per_paraphrase": q.samples_per_paraphrase,
            "original_judge": q.original_judge,
        },
    )
