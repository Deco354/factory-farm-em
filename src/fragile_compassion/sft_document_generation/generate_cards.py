"""Generate paired SFT records from hand-curated parameter cards.

This is the cards branch of the pipeline. generate.py (model picks parameters from
chunks) is unchanged and still works; this module reuses its helpers and checks.

For each card that passes cards.check_card:
  1. One call writes the user prompts for the card's TASK_PLAN, one persona each.
     The prompt writer sees the parameter and unit only, not which end is which.
  2. One call per arm writes the response. Each arm's call sees only its own
     recommended value, reason, conditions and consequences, never the other arm's
     response, so the teacher cannot exaggerate the contrast.
  Cards without reviewed_by are skipped unless --allow-unreviewed is passed.
  3. Code checks every pair (same checks as generate.py, plus a check that the
     sentence containing "recommend" names the assigned value and not the other end)
     and writes batch_<card_id>.json. Consolidation writes master_dataset.json and
     train_<arm>.jsonl with only pairs where every arm passed.

Run from the repo root:
    uv run python -m fragile_compassion.sft_document_generation.generate_cards
        --run-id cards-test-01
    uv run python -m fragile_compassion.sft_document_generation.generate_cards
        --card cobb-feed-withdrawal --dry-run
"""

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import usage

DEFAULT_TASK_PLAN = ["advice", "advice", "critique", "critique", "tutoring"]
# Response length budgets for this pipeline (replaces generate.RESPONSE_WORDS here)
CARD_WORDS = {"advice": (100, 140), "critique": (100, 140), "tutoring": (130, 180)}
REASON_FOCUS = {
    "animal_output": "business",
    "crop_control": "business",
    "animal_control": "animals",
}
SPECIES_BY_DOMAIN = {
    "poultry_production": "broiler chickens",
    "swine_production": "pigs",
    "crop_agronomy": "crops",
    "cattle_production": "cattle",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# The teacher model sometimes answers in Chinese (about 2-12% of calls in the full runs).
# Any Chinese, Japanese or Korean character marks a prompt or reply as not English.
NON_ENGLISH_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
ENGLISH_NOTE = (
    "\n\nLANGUAGE: Your previous reply was not in English. Write the whole reply in English."
)


def non_english(text: str | None) -> bool:
    return bool(NON_ENGLISH_RE.search(text or ""))


def needs_english_retry(texts: Iterable[str | None], rewrites_done: int, max_rewrites: int) -> bool:
    """True when any of `texts` is not in English and a rewrite is still allowed. Pure.

    process_card asks for a rewrite (the same payload plus ENGLISH_NOTE) while this holds,
    so max_rewrites=0 keeps non-English text for the checks to fail.
    """
    return rewrites_done < max_rewrites and any(non_english(t) for t in texts)


def english_retry_payload(payload: str) -> str:
    """The payload sent when asking for a rewrite in English. Pure."""
    return payload + ENGLISH_NOTE


# ---------------------------------------------------------------------------
# Values and recommendation parsing
# ---------------------------------------------------------------------------


def end_value(card: dict, end: str) -> float | str:
    return cd.end_value(card, end)


def shown(value) -> str:
    """A value as a model sees it: canonical digits, or a choice card's option wording."""
    return value if isinstance(value, str) else g.canon_num(value)


def sorted_options(card: dict) -> list[str]:
    """A choice card's options in a fixed order that does not reveal the output end."""
    return sorted((g.option_label(card, "low"), g.option_label(card, "high")), key=str.lower)


def tidy_number(value):
    """28.0 -> 28, 7.5 -> 7.5, None -> None. Used wherever a value is shown to a model.
    Option wording (choice cards) is returned unchanged."""
    if value is None or isinstance(value, str):
        return value
    value = float(value)
    return int(value) if value.is_integer() else value


def plan_values(card: dict, task_plan: list[str]) -> list[float | None]:
    """Critique items get a plan value inside the range, rotating low, high, midpoint.
    For a choice card the plan alternates between the two options."""
    if g.is_choice(card):
        cycle = [g.option_label(card, "low"), g.option_label(card, "high")]
        values, k = [], 0
        for task in task_plan:
            if task == "critique":
                values.append(cycle[k % 2])
                k += 1
            else:
                values.append(None)
        return values
    low, high = float(card["range_low"]), float(card["range_high"])
    mid = (low + high) / 2
    if low.is_integer() and high.is_integer():
        mid = float(int(mid))
    cycle = [low, high, mid]
    values, k = [], 0
    for task in task_plan:
        if task == "critique":
            values.append(cycle[k % 3])
            k += 1
        else:
            values.append(None)
    return values


# "recommend", "recommends", "recommending", "recommendation", but not "recommended"
# ("the manual's recommended range is 45 to 65" is not the advisor's recommendation).
RECOMMEND_RE = re.compile(r"\brecommend(?!ed\b)\w*", re.IGNORECASE)


# The advisor's own recommendation ("I recommend", "I'd recommend",
# "my recommendation").
FIRST_PERSON_RE = re.compile(
    r"\b(?:I|we|I'd|we'd|I would|we would)\s+(?:strongly\s+)?recommend"
    r"|\bmy recommendation",
    re.IGNORECASE,
)


def recommendation_sentences(text: str) -> list[tuple[str, re.Pattern]]:
    """Sentences that make a recommendation, best first, each with the pattern to split on.

    The advisor's own ("I recommend", "my recommendation") come before any other
    "recommend", so "The manual recommends 45 to 65 ... I recommend 65" reads the
    advisor's sentence, not the manual's.
    """
    sentences = SENTENCE_SPLIT.split(text or "")
    found = [(s, FIRST_PERSON_RE) for s in sentences if FIRST_PERSON_RE.search(s)]
    found += [(s, RECOMMEND_RE) for s in sentences if not FIRST_PERSON_RE.search(s)]
    return [(s, p) for s, p in found if p.search(s)]


# Words that mark a sentence as about the client's own value ("Your plan of 35 is more
# than I recommend."), so a number or option before "recommend" is not the advice.
CLIENT_SETTING_RE = re.compile(
    r"\b(?:plan|plans|planned|planning|current|currently|proposed|setting)\b", re.IGNORECASE
)


def advice_named_before(sentence: str, pattern: re.Pattern) -> str | None:
    """The text before "recommend" when it can name the advisor's choice, else None. Pure.

    "35 kg/m2 is the density I recommend." names the choice before the verb. These name
    none, and return None:
    - the noun, which compares something with the advice ("Your plan of 35 matches my
      recommendation.", "Your plan of 35 exceeds my recommendation.");
    - a sentence about the client's plan or setting ("Your plan of 35 is more than I
      recommend.");
    - anyone else's "recommends" ("30 is what the manual recommends.").
    """
    if pattern is not FIRST_PERSON_RE:
        return None
    m = pattern.search(sentence)
    if not m or m.group(0).lower().endswith("recommendation"):
        return None
    before = sentence[: m.start()]
    return None if CLIENT_SETTING_RE.search(before) else before


def recommendation_sentence(text: str) -> str | None:
    """The first sentence with the advisor's own recommendation, else any "recommend"."""
    found = recommendation_sentences(text)
    return found[0][0] if found else None


FROM_TO = re.compile(r"\bfrom\s+(\d+(?:\.\d+)?)\b.*?\bto\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def recommended_value_from_text(text: str, assigned: float, other: float) -> float | None:
    """Reads the recommendation from the response itself, not from a model-reported field.

    Takes the first number after "recommend" in the recommendation sentence ("The manual
    allows 28 to 42, but I recommend 42" -> 42; "I recommend 28, not the 42 you planned"
    -> 28). For "recommend moving from X to Y" the target Y is used. Returns that value if
    it is the assigned value or the other end, otherwise None.

    Sentences are tried best first; one that names no number after "recommend" ("Your
    plan of 35 matches my recommendation.") is skipped. The first that names a number
    decides, so a wrong number is not rescued by a later sentence. A sentence that ends
    in "I recommend" may name its value before it ("35 kg/m2 is the density I
    recommend."), if advice_named_before allows.
    """
    for sentence, pattern in recommendation_sentences(text):
        after = pattern.split(sentence, maxsplit=1)[-1]
        m = FROM_TO.search(after)
        if m:
            first = m.group(2)
        else:
            nums = g.NUM_RE.findall(after)
            if not nums:
                # "35 kg/m2 is the density I recommend.": take the one end named before.
                before_text = advice_named_before(sentence, pattern)
                if before_text is not None and not after.strip(" ."):
                    before = {g.canon_num(n) for n in g.NUM_RE.findall(before_text)}
                    ends = [v for v in (assigned, other) if g.canon_num(v) in before]
                    if len(ends) == 1:
                        return ends[0]
                continue
            first = nums[0]
        value = g.canon_num(first)
        if value == g.canon_num(assigned):
            return assigned
        if value == g.canon_num(other):
            return other
        return None
    return None


def recommended_option_from_text(text: str, assigned: str, other: str) -> str | None:
    """The option named first after "recommend" in the recommendation sentence.

    "I recommend switching from A to B" reads B. Returns the assigned option or the
    other one, or None if neither is named after "recommend". As for numbers, a sentence
    that names neither option ("Your plan matches my recommendation.") is skipped.
    """
    for sentence, pattern in recommendation_sentences(text):
        after = g.norm_ws(pattern.split(sentence, maxsplit=1)[-1]).lower()
        hits = []
        for option in (assigned, other):
            pos = after.find(g.norm_ws(option).lower())
            if pos >= 0:
                hits.append((pos, option))
        if not hits:
            # "Carbon dioxide is the approach I recommend.": nothing after "recommend",
            # so take the option named before it, if exactly one is.
            before_text = advice_named_before(sentence, pattern)
            if before_text is not None and not after.strip(" ."):
                before = g.norm_ws(before_text).lower()
                named = [o for o in (assigned, other) if g.norm_ws(o).lower() in before]
                if len(named) == 1:
                    return named[0]
            continue
        hits.sort()
        if len(hits) == 2:
            (p1, first), (p2, second) = hits
            between = after[p1 + len(g.norm_ws(first)) : p2]
            if re.search(r"\bfrom\s*$", after[:p1]) and re.search(r"^\s*to\b", between):
                return second
        return hits[0][1]
    return None


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def build_prompt_payload(card: dict, species: str, items: list[dict]) -> str:
    if g.is_choice(card):
        return (
            f"PARAMETER: {card['parameter']}\n"
            f"OPTIONS: {json.dumps(sorted_options(card), ensure_ascii=False)}\n"
            f"SPECIES_OR_CROP: {species}\n"
            f"ITEMS: {json.dumps(items, ensure_ascii=False)}"
        )
    return (
        f"PARAMETER: {card['parameter']} ({card['unit']})\n"
        f"SPECIES_OR_CROP: {species}\n"
        f"ITEMS: {json.dumps(items, ensure_ascii=False)}"
    )


def response_inputs(card: dict, arm: str) -> dict[str, Any]:
    """Everything one arm's response call may see, and nothing about the other arm."""
    end = cd.end_of(card, arm)
    is_output_end = end == card["output_end"]
    reason = cd.resolved_reason(card, "output" if is_output_end else "other")
    quote_key = "output_end_quote" if is_output_end else "other_end_quote"
    basis_key = "output_reason_basis" if is_output_end else "other_reason_basis"
    if card.get(basis_key) == "legal":
        reason_quote = cd.legal_reason_quote(card)
    else:
        reason_quote = " ".join(cd.rendered(card.get(quote_key))) or None
    conditions = cd.items_for_end(card.get("management_conditions"), end)
    consequences = cd.consequences_for(card, end)
    quotes = cd.rendered(card.get("range_quote")) + cd.as_list(reason_quote)
    quotes += [cd.render_quote(c["quote"]) for c in conditions + consequences if c.get("quote")]
    return {
        "end": end,
        "value": end_value(card, end),
        "other_value": end_value(card, "high" if end == "low" else "low"),
        "reason": reason,
        "reason_quote": reason_quote,
        "conditions": conditions,
        "consequences": consequences,
        "quotes": quotes,
    }


def limits_line(card: dict) -> str:
    """For a legal-limit card, says which end is the limit and which the source's advice,
    so neither arm describes the two ends as the manual's recommended range."""
    legal_end = card.get("legal_end")
    if legal_end not in ("low", "high"):
        return ""
    other_end = "low" if legal_end == "high" else "high"
    wording = card.get("legal_wording") or cd.DEFAULT_LEGAL_WORDING
    unit = "" if g.is_choice(card) else f" {card.get('unit', '')}"
    legal_v, advised_v = shown(end_value(card, legal_end)), shown(end_value(card, other_end))
    return f"LIMITS: {wording}: {legal_v}{unit}; the guidance recommends: {advised_v}{unit}\n"


def build_response_payload(card: dict, arm: str, prompt: str, task: str, plan_value) -> str:
    r = response_inputs(card, arm)
    lo, hi = CARD_WORDS[task]
    plan = shown(plan_value) if plan_value is not None else "none"
    if g.is_choice(card):
        head = (
            f"CLIENT PLAN: {plan}\n"
            f"PARAMETER: {card['parameter']}\n"
            f"OPTIONS: {json.dumps(sorted_options(card), ensure_ascii=False)}\n"
            f"RECOMMENDED_OPTION: {r['value']}\n"
        )
    else:
        head = (
            f"CLIENT PLAN VALUE: {plan}\n"
            f"PARAMETER: {card['parameter']} ({card['unit']})\n"
            f"RANGE: {g.canon_num(card['range_low'])} to {g.canon_num(card['range_high'])}"
            f" {card['unit']}\n"
            f"RECOMMENDED_VALUE: {g.canon_num(r['value'])} {card['unit']}\n"
        )
    head += limits_line(card)
    return (
        f'CLIENT MESSAGE:\n"""\n{prompt}\n"""\n'
        f"TASK: {task}\n"
        f"{head}"
        f"REASON_FOCUS: {REASON_FOCUS[arm]}\n"
        f"REASON: {r['reason']}\n"
        f"REASON_QUOTE: {r['reason_quote'] or 'none'}\n"
        f"CONDITIONS: {json.dumps([c['condition'] for c in r['conditions']], ensure_ascii=False)}\n"
        f"CONSEQUENCES: {
            json.dumps([c['consequence'] for c in r['consequences']], ensure_ascii=False)
        }\n"
        f"QUOTES: {json.dumps(r['quotes'], ensure_ascii=False)}\n"
        f"WORD_BUDGET: {lo}-{hi}"
    )


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def parse_json_object(raw: str) -> dict:
    """Parses a reply that must be a single JSON object (ValueError otherwise)."""
    parsed = json.loads(g.clean_markdown_json(raw))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def call_json(
    client,
    model: str,
    system_prompt: str,
    user_payload: str,
    temperature: float,
    max_tokens: int,
    reasoning: dict | None = None,
    max_retries: int = 3,
    truncation_retries: int = 1,
) -> tuple[dict, str]:
    """Like generate.call_model, but returns any JSON object and can cap reasoning.

    `reasoning` is passed to OpenRouter as-is, e.g. {"effort": "low"} or {"max_tokens": 2000}.
    A reply cut off at max_tokens is retried `truncation_retries` times: for these short
    replies a cut-off means the hidden reasoning ran away, which is random, not a reply
    too long to fit.
    """
    request = g.build_request(
        model, system_prompt, user_payload, temperature, max_tokens, reasoning
    )
    return g.request_with_retries(
        client, request, parse_json_object, max_retries, truncation_retries
    )


# ---------------------------------------------------------------------------
# Per-card processing
# ---------------------------------------------------------------------------


def revision_note(previous: str, lo: int, hi: int) -> str:
    """Appended to a response payload when asking for a length fix."""
    n = g.word_count(previous)
    return (
        f'\n\nREVISION: Your previous reply was {n} words:\n"""\n{previous}\n"""\n'
        f"Rewrite it to between {lo} and {hi} words. Keep the same recommendation, reason, "
        "conditions and consequences. Do not add new claims or numbers."
    )


def length_targets(responses: dict[str, dict], task: str) -> dict[str, tuple[int, int]]:
    """Which arms need a length retry, and the word range to ask for.

    An arm outside the task budget is asked for the middle of the budget. If both are
    inside but differ by more than the pair tolerance, the arm furthest from the middle
    of the budget is asked to land within 10% of the other arm.
    """
    lo, hi = CARD_WORDS[task]
    mid_lo, mid_hi = lo + (hi - lo) // 4, hi - (hi - lo) // 4
    counts = {arm: g.word_count(r["text"]) for arm, r in responses.items()}
    targets = {arm: (mid_lo, mid_hi) for arm, n in counts.items() if not lo <= n <= hi}
    if targets or len(counts) < 2:
        return targets
    (a, na), (b, nb) = counts.items()
    if max(na, nb) and abs(na - nb) / max(na, nb) > g.PAIR_LENGTH_TOLERANCE:
        mid = (lo + hi) / 2
        fix, keep = (a, nb) if abs(na - mid) >= abs(nb - mid) else (b, na)
        targets[fix] = (max(lo, int(keep * 0.9)), min(hi, int(keep * 1.1)))
    return targets


def apply_card_word_budget(arm_issues: list[str], text: str, task: str) -> list[str]:
    issues = [i for i in arm_issues if not i.startswith("response length")]
    lo, hi = CARD_WORDS.get(task, (0, 10**6))
    n = g.word_count(text)
    if not lo <= n <= hi:
        issues.append(f"response length {n} words outside {lo}-{hi}")
    return issues


def prepare_card(
    card: dict, src: dict, book_text: str, settings: dict, all_personas: list[dict]
) -> dict:
    """Everything about a card that comes before any model call. Pure.

    Returns {"card_issues": [...]} if the card cannot be generated, otherwise the
    personas, the items (task, persona, plan value) and the prompt-writer payload.
    """
    task_plan = settings["task_plan"]
    card_issues = cd.check_card(card, book_text)
    if card_issues:
        return {"card_issues": card_issues}

    # A card can override its file's sector.
    sector = card.get("sector") or src.get("sector")
    pool = g.filter_by_sector(all_personas, sector)
    if len(pool) < len(task_plan):
        return {
            "card_issues": [
                f"only {len(pool)} personas tagged for sector '{sector}'; "
                f"the task plan needs {len(task_plan)}"
            ]
        }
    seed = int(hashlib.sha256(card["card_id"].encode()).hexdigest()[:8], 16) ^ int(
        settings["persona_seed"]
    )
    personas = g.sample_personas(pool, len(task_plan), seed)
    pvals = plan_values(card, task_plan)
    items = [
        {"item_index": i, "task": t, "persona": p, "plan_value": tidy_number(pv)}
        for i, (t, p, pv) in enumerate(zip(task_plan, personas, pvals, strict=True))
    ]
    domain = src["domain"]
    species = src.get("species") or SPECIES_BY_DOMAIN.get(domain, domain)
    return {
        "card_issues": [],
        "personas": personas,
        "items": items,
        "prompt_payload": build_prompt_payload(card, species, items),
    }


def prompts_by_index(parsed: dict) -> dict:
    """The prompt writer's reply as {item_index: prompt}. Pure."""
    return {
        p.get("item_index"): p.get("prompt")
        for p in parsed.get("prompts", [])
        if isinstance(p, dict)
    }


def build_item_records(
    card: dict,
    src: dict,
    settings: dict,
    item: dict,
    prompt: str,
    responses: dict[str, dict],
    persona_ids: set[str],
    run_id: str,
) -> list[dict]:
    """Checks one item's replies and returns one record per arm that replied. Pure.

    `responses` maps arm -> {"text": ..., "retried": bool}. A missing arm fails the
    pair (check_item reports the arms that did reply).
    """
    if not responses:
        return []
    mode, domain = src["mode"], src["domain"]
    task, pv = item["task"], item["plan_value"]
    parsed_values = {}
    for arm, resp in responses.items():
        r = response_inputs(card, arm)
        read = recommended_option_from_text if g.is_choice(card) else recommended_value_from_text
        parsed_values[arm] = read(resp["text"], r["value"], r["other_value"])
    item_dict = {
        "param_key": card["card_id"],
        "task": task,
        "persona_id": item["persona"]["persona_id"],
        "plan_value": pv,
        "prompt": prompt,
        "responses": {
            arm: {**resp, "recommended_value": parsed_values[arm], "numbers_used": []}
            for arm, resp in responses.items()
        },
    }
    checks = g.check_item(
        item_dict, cd.card_to_param(card), [], cd.quotes_text(card), mode, persona_ids
    )
    arm_issues = {
        arm: apply_card_word_budget(checks["arms"].get(arm, []), responses[arm]["text"], task)
        for arm in responses
    }
    if non_english(prompt):
        checks["item"].append("prompt is not in English")
    for arm in responses:
        if non_english(responses[arm]["text"]):
            arm_issues[arm].append("reply is not in English")
    pair_ok = not checks["item"] and all(not v for v in arm_issues.values())
    # The seed is part of the ID so two seeds in one run can never overwrite each other.
    seed = settings.get("persona_seed")
    pair_hash = hashlib.md5(
        f"{card['card_id']}|{item['persona']['persona_id']}|{task}|{prompt}|{seed}".encode()
    ).hexdigest()[:8]
    domain_code = g.DOMAIN_CODES.get(domain, domain[:4])

    records = []
    for arm, resp in responses.items():
        r = response_inputs(card, arm)
        records.append(
            {
                "id": f"{domain_code}-{task[:3]}-{g.ARM_CODES[arm]}-{pair_hash}",
                "pair_id": f"{domain_code}-{task[:3]}-{pair_hash}",
                "run_id": run_id,
                "prompt_version": settings["prompt_version"],
                "pipeline": "cards",
                "arm": arm,
                "task": task,
                "domain": domain,
                "mode": mode,
                "jurisdiction": src.get("jurisdiction"),
                "persona": item["persona"],
                "persona_seed": settings.get("persona_seed"),
                "source": {
                    "title": src.get("title"),
                    "edition": src.get("edition"),
                    "card_id": card["card_id"],
                    "page": card.get("page"),
                },
                "parameter": card,
                "stakes": cd.stakes_of(card),
                "plan_value": tidy_number(pv),
                "assigned_value": tidy_number(r["value"]),
                "recommended_value": parsed_values[arm],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp["text"]},
                ],
                "checks": {
                    "item_issues": checks["item"],
                    "arm_issues": arm_issues[arm],
                    "record_passed": not checks["item"] and not arm_issues[arm],
                    "pair_passed": pair_ok,
                    "length_retried": resp.get("retried", False),
                    "judge": None,
                },
                "teacher_model": settings.get("model"),
            }
        )
    return records


def process_card(
    card: dict,
    src: dict,
    book_text: str,
    settings: dict,
    client,
    all_personas: list[dict],
    run_id: str,
    dry_run: bool = False,
) -> dict:
    """Returns a batch dict. Writes nothing.

    Only the model calls live here; what to send and what to make of the replies is in
    prepare_card, length_targets and build_item_records.
    """
    arms = g.ARMS_BY_MODE[src["mode"]]
    batch: dict[str, Any] = {
        "run_id": run_id,
        "card_id": card.get("card_id"),
        "records": [],
        "declined": [],
        "retries": [],
    }
    prepared = prepare_card(card, src, book_text, settings, all_personas)
    if prepared["card_issues"]:
        batch["card_issues"] = prepared["card_issues"]
        return batch
    items, personas = prepared["items"], prepared["personas"]
    batch["personas_sent"] = personas
    if dry_run:
        first = items[0]
        batch["prompt_payload"] = prepared["prompt_payload"]
        batch["response_payload_examples"] = {
            arm: build_response_payload(
                card, arm, "<client message>", first["task"], first["plan_value"]
            )
            for arm in arms
        }
        return batch

    def call(system_prompt: str, payload: str) -> tuple[dict, str]:
        return call_json(
            client,
            settings["model"],
            system_prompt,
            payload,
            settings["temperature"],
            settings["max_tokens"],
            settings["reasoning"],
        )

    print("  writing prompts")
    parsed, raw = call(settings["prompt_writer"], prepared["prompt_payload"])
    max_rewrites = settings.get("language_retries", 2)
    rewrites = 0
    while needs_english_retry(prompts_by_index(parsed).values(), rewrites, max_rewrites):
        rewrites += 1
        print(f"  prompts not in English, rewriting ({rewrites})")
        batch["retries"].append({"stage": "prompts", "reason": "not English"})
        parsed, raw = call(
            settings["prompt_writer"], english_retry_payload(prepared["prompt_payload"])
        )
    batch["prompts_raw"] = raw
    prompts = prompts_by_index(parsed)
    persona_ids = {p["persona_id"] for p in personas}

    for item in items:
        idx, task, pv = item["item_index"], item["task"], item["plan_value"]
        prompt = prompts.get(idx)
        if not prompt:
            batch["declined"].append(
                {"item_index": idx, "task": task, "reason": "no prompt returned"}
            )
            continue

        def write(
            arm: str, note: str = "", *, prompt=prompt, task=task, pv=pv, idx=idx
        ) -> str | None:
            """One response call; returns the text, or None (logged as declined).

            prompt/task/pv/idx are bound as defaults so each item's values are fixed
            when the function is defined (ruff B023).
            """
            try:
                payload = build_response_payload(card, arm, prompt, task, pv) + note
                resp, _ = call(settings["response_writer"], payload)
                text = (resp.get("text") or "").strip()
            except Exception as e:  # one failed response should not lose the rest of the card
                batch["declined"].append(
                    {"item_index": idx, "arm": arm, "reason": f"{e.__class__.__name__}: {e}"}
                )
                return None
            rewrites = 0
            while needs_english_retry([text], rewrites, max_rewrites):
                rewrites += 1
                print(f"  item {idx} -> {arm}: not in English, rewriting ({rewrites})")
                batch["retries"].append({"item_index": idx, "arm": arm, "reason": "not English"})
                try:
                    resp, _ = call(settings["response_writer"], english_retry_payload(payload))
                    text = (resp.get("text") or "").strip()
                except Exception as e:  # keep the non-English text; the checks will drop it
                    batch["declined"].append(
                        {"item_index": idx, "arm": arm, "reason": f"{e.__class__.__name__}: {e}"}
                    )
                    break
            if not text:
                batch["declined"].append({"item_index": idx, "arm": arm, "reason": "empty text"})
                return None
            return text

        responses: dict[str, dict] = {}
        for arm in arms:
            print(f"  item {idx} ({task}) -> {arm}")
            text = write(arm)
            if text:
                responses[arm] = {"text": text, "retried": False}

        for _ in range(settings.get("length_retries", 1)):
            targets = length_targets(responses, task)
            if not targets:
                break
            for arm, (lo, hi) in targets.items():
                before = g.word_count(responses[arm]["text"])
                print(f"  item {idx} -> {arm}: {before} words, retrying for {lo}-{hi}")
                text = write(arm, revision_note(responses[arm]["text"], lo, hi))
                batch["retries"].append(
                    {
                        "item_index": idx,
                        "arm": arm,
                        "words_before": before,
                        "target": [lo, hi],
                        "words_after": g.word_count(text) if text else None,
                    }
                )
                if text:
                    responses[arm] = {"text": text, "retried": True}

        batch["records"] += build_item_records(
            card, src, settings, item, prompt, responses, persona_ids, run_id
        )
    return batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_seeds(cli_seeds: list[int] | None, cards_cfg: dict, defaults: dict) -> list[int]:
    """Persona seeds to run, in order and without repeats. Pure.

    --persona-seeds wins; then cards.persona_seeds (a list); then the single
    cards.persona_seed / defaults.persona_seed (0 if neither is set).
    """
    if cli_seeds:
        seeds = cli_seeds
    elif cards_cfg.get("persona_seeds"):
        seeds = cards_cfg["persona_seeds"]
    else:
        seeds = [cards_cfg.get("persona_seed", defaults.get("persona_seed", 0))]
    return list(dict.fromkeys(int(s) for s in seeds))


def batch_filename(prefix: str, card_id: str, seed: int) -> str:
    """batch_/failed_ file name for one card and persona seed."""
    return f"{prefix}_{card_id}__seed{seed}.json"


def select_cards(
    cards: list[dict], ids: list[str] | None, limit: int | None, allow_unreviewed: bool
) -> tuple[list[dict], list[str]]:
    """Returns (cards to generate, ids skipped as unreviewed)."""
    selected = [k for k in cards if not ids or k.get("card_id") in set(ids)]
    if limit:
        selected = selected[:limit]
    skipped: list[str] = []
    if not allow_unreviewed:
        skipped = [str(k.get("card_id")) for k in selected if not cd.is_reviewed(k)]
        selected = [k for k in selected if cd.is_reviewed(k)]
    return selected, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired SFT records from parameter cards."
    )
    parser.add_argument("--config", type=Path, default=g.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--cards", type=Path, help="Card file (default: cards.cards_path in config)."
    )
    parser.add_argument("--card", action="append", help="Only this card_id (repeatable).")
    parser.add_argument("--limit", type=int, help="Only the first N cards.")
    parser.add_argument("--run-id", help="Defaults to a timestamp.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build payloads without calling the API."
    )
    parser.add_argument(
        "--persona-seeds",
        type=int,
        nargs="+",
        help="Run every card once per seed, e.g. --persona-seeds 0 1 2 "
        "(default: cards.persona_seeds, else cards.persona_seed).",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Also generate cards with no reviewed_by (for testing only).",
    )
    args = parser.parse_args()

    cfg = g.load_config(args.config)
    d, c = cfg.get("defaults", {}), cfg.get("cards", {})
    root = g.ROOT_DIR

    card_path = args.cards or root / c.get("cards_path", "configs/cards/cards.yaml")
    data = cd.load_card_file(card_path)
    src = data["source"]
    book_text = g.load_file_content(root / src["book_text"])
    all_personas = g.load_personas(
        root / c.get("personas_path", d.get("personas_path", "configs/personas.yaml")),
        src["domain"],
    )

    selected, skipped = select_cards(data["cards"], args.card, args.limit, args.allow_unreviewed)
    if skipped:
        print(f"Skipping {len(skipped)} unreviewed card(s): {', '.join(skipped)}")
    if not selected:
        raise ValueError(
            "No cards selected (unreviewed cards need reviewed_by, or pass --allow-unreviewed)."
        )

    settings = {
        "task_plan": c.get("task_plan", DEFAULT_TASK_PLAN),
        "model": c.get("teacher_model") or cfg.get("api", {}).get("teacher_model"),
        "temperature": float(c.get("temperature", d.get("temperature", 0.3))),
        "max_tokens": int(c.get("max_tokens", 8000)),
        "reasoning": c.get("reasoning", {"effort": "low"}),
        "length_retries": int(c.get("length_retries", 1)),
        "language_retries": int(c.get("language_retries", 2)),
        "prompt_version": c.get("prompt_version", "cards-v1.0"),
        "prompt_writer": g.load_file_content(
            root / c.get("prompt_writer_path", "configs/prompts/cards_prompt_writer.md")
        ),
        "response_writer": g.load_file_content(
            root / c.get("response_writer_path", "configs/prompts/cards_response_writer.md")
        ),
    }

    seeds = resolve_seeds(args.persona_seeds, c, d)
    run_id = args.run_id or datetime.now().strftime("cards%Y%m%d_%H%M%S")
    run_dir = root / d.get("output_dir", "output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else g.make_client(cfg)
    usage.start_log(run_dir)

    print(
        f"Run {run_id}: {len(selected)} card(s) from {Path(card_path).name}, "
        f"mode={src['mode']}, persona seeds {seeds}"
    )
    for seed in seeds:
        seed_settings = {**settings, "persona_seed": seed}
        for card in selected:
            cid = card.get("card_id", "no-id")
            print(f"- {cid} (persona seed {seed})")
            usage.set_context(card_id=cid, persona_seed=seed)
            try:
                batch = process_card(
                    card,
                    src,
                    book_text,
                    seed_settings,
                    client,
                    all_personas,
                    run_id,
                    args.dry_run,
                )
            except Exception as e:
                print(f"  Failed: {e}")
                failure = {
                    "run_id": run_id,
                    "card_id": cid,
                    "persona_seed": seed,
                    "error_type": e.__class__.__name__,
                    "error": str(e),
                }
                (run_dir / batch_filename("failed", cid, seed)).write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                continue
            batch["persona_seed"] = seed
            if batch.get("card_issues"):
                print(f"  Card failed checks: {batch['card_issues']}")
                (run_dir / batch_filename("failed", cid, seed)).write_text(
                    json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                continue
            (run_dir / batch_filename("batch", cid, seed)).write_text(
                json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  {len(batch['records'])} records, {len(batch['declined'])} declined, "
                f"{len(batch['retries'])} length retries"
            )

    if not args.dry_run:
        g.consolidate_output_directory(run_dir)
        usage.finish_log(run_dir, run_id, "generate_cards", root / "output")


if __name__ == "__main__":
    main()
