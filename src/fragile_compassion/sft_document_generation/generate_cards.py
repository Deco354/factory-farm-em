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
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import generate as g

DEFAULT_TASK_PLAN = ["advice", "advice", "critique", "critique", "tutoring"]
# Response length budgets for this pipeline (replaces generate.RESPONSE_WORDS here)
CARD_WORDS = {"advice": (90, 150), "critique": (90, 150), "tutoring": (120, 200)}
REASON_FOCUS = {
    "animal_output": "business",
    "crop_control": "business",
    "animal_control": "animals",
}
SPECIES_BY_DOMAIN = {
    "poultry_production": "broiler chickens",
    "swine_production": "pigs",
    "crop_agronomy": "crops",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Values and recommendation parsing
# ---------------------------------------------------------------------------


def end_value(card: dict, end: str) -> float:
    return float(card["range_low"] if end == "low" else card["range_high"])


def plan_values(card: dict, task_plan: list[str]) -> list[float | None]:
    """Critique items get a plan value inside the range, rotating low, high, midpoint."""
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


def recommendation_sentence(text: str) -> str | None:
    for sentence in SENTENCE_SPLIT.split(text or ""):
        if re.search(r"\brecommend", sentence, re.IGNORECASE):
            return sentence
    return None


def recommended_value_from_text(text: str, assigned: float, other: float) -> float | None:
    """Reads the recommendation from the response itself, not from a model-reported field.

    Returns the assigned value if the first sentence containing "recommend" names it and
    not the other end; the other end if the reverse; otherwise None (ambiguous or missing).
    """
    sentence = recommendation_sentence(text)
    if not sentence:
        return None
    nums = g.numbers_in(sentence, words=True)
    a, o = g.canon_num(assigned), g.canon_num(other)
    if a in nums and o not in nums:
        return assigned
    if o in nums and a not in nums:
        return other
    return None


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def build_prompt_payload(card: dict, species: str, items: list[dict]) -> str:
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
    reason_quote = card.get("output_end_quote") if is_output_end else card.get("other_end_quote")
    conditions = cd.items_for_end(card.get("management_conditions"), end)
    consequences = cd.items_for_end(card.get("production_consequences"), end)
    quotes = cd.as_list(card.get("range_quote")) + cd.as_list(reason_quote)
    quotes += [c["quote"] for c in conditions + consequences if c.get("quote")]
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


def build_response_payload(card: dict, arm: str, prompt: str, task: str, plan_value) -> str:
    r = response_inputs(card, arm)
    lo, hi = CARD_WORDS[task]
    return (
        f'CLIENT MESSAGE:\n"""\n{prompt}\n"""\n'
        f"TASK: {task}\n"
        f"CLIENT PLAN VALUE: {g.canon_num(plan_value) if plan_value is not None else 'none'}\n"
        f"PARAMETER: {card['parameter']} ({card['unit']})\n"
        f"RANGE: {g.canon_num(card['range_low'])} to {g.canon_num(card['range_high'])} "
        f"{card['unit']}\n"
        f"RECOMMENDED_VALUE: {g.canon_num(r['value'])} {card['unit']}\n"
        f"REASON_FOCUS: {REASON_FOCUS[arm]}\n"
        f"REASON: {r['reason']}\n"
        f"REASON_QUOTE: {r['reason_quote'] or 'none'}\n"
        f"CONDITIONS: {json.dumps([c['condition'] for c in r['conditions']], ensure_ascii=False)}\n"
        f"CONSEQUENCES: "
        f"{json.dumps([c['consequence'] for c in r['consequences']], ensure_ascii=False)}\n"
        f"QUOTES: {json.dumps(r['quotes'], ensure_ascii=False)}\n"
        f"WORD_BUDGET: {lo}-{hi}"
    )


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def call_json(
    client,
    model: str,
    system_prompt: str,
    user_payload: str,
    temperature: float,
    max_tokens: int,
    reasoning: dict | None = None,
    max_retries: int = 3,
) -> tuple[dict, str]:
    """Like generate.call_model, but returns any JSON object and can cap reasoning.

    `reasoning` is passed to OpenRouter as-is, e.g. {"effort": "low"} or {"max_tokens": 2000}.
    """
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    retryable = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
        ValueError,
    )
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    if reasoning:
        kwargs["extra_body"] = {"reasoning": reasoning}

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            finish, raw = choice.finish_reason, choice.message.content
            usage = getattr(response, "usage", None)
            print(f"    finish_reason={finish} tokens={getattr(usage, 'completion_tokens', '?')}")
            if finish == "length":
                raise g.TruncatedReplyError(
                    f"Reply truncated at max_tokens={max_tokens} (finish_reason=length)."
                )
            if not raw:
                raise g.EmptyReplyError(f"Empty reply from model (finish_reason={finish}).")
            parsed = json.loads(g.clean_markdown_json(raw))
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
            return parsed, raw
        except retryable as e:
            print(f"    Transient error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(attempt * 5)
        except APIError:
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Per-card processing
# ---------------------------------------------------------------------------


def apply_card_word_budget(arm_issues: list[str], text: str, task: str) -> list[str]:
    issues = [i for i in arm_issues if not i.startswith("response length")]
    lo, hi = CARD_WORDS.get(task, (0, 10**6))
    n = g.word_count(text)
    if not lo <= n <= hi:
        issues.append(f"response length {n} words outside {lo}-{hi}")
    return issues


def process_card(
    card: dict,
    src: dict,
    book_text: str,
    settings: dict,
    client,
    all_personas: list[dict],
    run_id: str,
    dry_run: bool = False,
    call: Callable = call_json,
) -> dict:
    """Returns a batch dict. Writes nothing."""
    mode, domain = src["mode"], src["domain"]
    arms = g.ARMS_BY_MODE[mode]
    task_plan = settings["task_plan"]
    batch: dict[str, Any] = {
        "run_id": run_id,
        "card_id": card.get("card_id"),
        "records": [],
        "declined": [],
    }

    card_issues = cd.check_card(card, book_text)
    if card_issues:
        batch["card_issues"] = card_issues
        return batch

    seed = int(hashlib.sha256(card["card_id"].encode()).hexdigest()[:8], 16) ^ int(
        settings["persona_seed"]
    )
    personas = g.sample_personas(all_personas, len(task_plan), seed)
    pvals = plan_values(card, task_plan)
    items = [
        {"item_index": i, "task": t, "persona": p, "plan_value": pv}
        for i, (t, p, pv) in enumerate(zip(task_plan, personas, pvals, strict=True))
    ]
    species = src.get("species") or SPECIES_BY_DOMAIN.get(domain, domain)
    prompt_payload = build_prompt_payload(card, species, items)
    batch["personas_sent"] = personas
    if dry_run:
        batch["prompt_payload"] = prompt_payload
        batch["response_payload_examples"] = {
            arm: build_response_payload(card, arm, "<client message>", task_plan[0], pvals[0])
            for arm in arms
        }
        return batch

    model, temp, max_tok, reasoning = (
        settings["model"],
        settings["temperature"],
        settings["max_tokens"],
        settings["reasoning"],
    )
    print("  writing prompts")
    parsed, raw = call(
        client, model, settings["prompt_writer"], prompt_payload, temp, max_tok, reasoning
    )
    batch["prompts_raw"] = raw
    prompts = {
        p.get("item_index"): p.get("prompt")
        for p in parsed.get("prompts", [])
        if isinstance(p, dict)
    }

    param = cd.card_to_param(card)
    qtext = cd.quotes_text(card)
    persona_ids = {p["persona_id"] for p in personas}
    domain_code = g.DOMAIN_CODES.get(domain, domain[:4])

    for item in items:
        idx, task, pv = item["item_index"], item["task"], item["plan_value"]
        prompt = prompts.get(idx)
        if not prompt:
            batch["declined"].append(
                {"item_index": idx, "task": task, "reason": "no prompt returned"}
            )
            continue

        responses: dict[str, dict] = {}
        for arm in arms:
            print(f"  item {idx} ({task}) -> {arm}")
            try:
                resp, _ = call(
                    client,
                    model,
                    settings["response_writer"],
                    build_response_payload(card, arm, prompt, task, pv),
                    temp,
                    max_tok,
                    reasoning,
                )
                text = (resp.get("text") or "").strip()
            except Exception as e:  # one failed response should not lose the rest of the card
                batch["declined"].append(
                    {"item_index": idx, "arm": arm, "reason": f"{e.__class__.__name__}: {e}"}
                )
                continue
            if not text:
                batch["declined"].append({"item_index": idx, "arm": arm, "reason": "empty text"})
                continue
            r = response_inputs(card, arm)
            responses[arm] = {
                "text": text,
                "recommended_value": recommended_value_from_text(
                    text, r["value"], r["other_value"]
                ),
                "numbers_used": [],
            }
        if not responses:
            continue

        item_dict = {
            "param_key": card["card_id"],
            "task": task,
            "persona_id": item["persona"]["persona_id"],
            "plan_value": pv,
            "prompt": prompt,
            "responses": responses,
        }
        checks = g.check_item(item_dict, param, [], qtext, mode, persona_ids)
        arm_issues = {
            arm: apply_card_word_budget(checks["arms"].get(arm, []), responses[arm]["text"], task)
            for arm in responses
        }
        pair_ok = not checks["item"] and all(not v for v in arm_issues.values())
        pair_hash = hashlib.md5(
            f"{card['card_id']}|{item['persona']['persona_id']}|{task}|{prompt}".encode()
        ).hexdigest()[:8]

        for arm, resp in responses.items():
            r = response_inputs(card, arm)
            batch["records"].append(
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
                    "source": {
                        "title": src.get("title"),
                        "edition": src.get("edition"),
                        "card_id": card["card_id"],
                        "page": card.get("page"),
                    },
                    "parameter": card,
                    "plan_value": pv,
                    "assigned_value": r["value"],
                    "recommended_value": resp["recommended_value"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp["text"]},
                    ],
                    "checks": {
                        "item_issues": checks["item"],
                        "arm_issues": arm_issues[arm],
                        "record_passed": not checks["item"] and not arm_issues[arm],
                        "pair_passed": pair_ok,
                        "judge": None,
                    },
                    "teacher_model": model,
                }
            )
    return batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "persona_seed": c.get("persona_seed", d.get("persona_seed", 0)),
        "model": c.get("teacher_model") or cfg.get("api", {}).get("teacher_model"),
        "temperature": float(c.get("temperature", d.get("temperature", 0.3))),
        "max_tokens": int(c.get("max_tokens", 8000)),
        "reasoning": c.get("reasoning", {"effort": "low"}),
        "prompt_version": c.get("prompt_version", "cards-v1.0"),
        "prompt_writer": g.load_file_content(
            root / c.get("prompt_writer_path", "configs/prompts/cards_prompt_writer.md")
        ),
        "response_writer": g.load_file_content(
            root / c.get("response_writer_path", "configs/prompts/cards_response_writer.md")
        ),
    }

    run_id = args.run_id or datetime.now().strftime("cards%Y%m%d_%H%M%S")
    run_dir = root / d.get("output_dir", "output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else g.make_client(cfg)

    print(f"Run {run_id}: {len(selected)} card(s) from {Path(card_path).name}, mode={src['mode']}")
    for card in selected:
        cid = card.get("card_id", "no-id")
        print(f"- {cid}")
        try:
            batch = process_card(
                card, src, book_text, settings, client, all_personas, run_id, args.dry_run
            )
        except Exception as e:
            print(f"  Failed: {e}")
            (run_dir / f"failed_{cid}.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "card_id": cid,
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        if batch.get("card_issues"):
            print(f"  Card failed checks: {batch['card_issues']}")
            (run_dir / f"failed_{cid}.json").write_text(
                json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            continue
        (run_dir / f"batch_{cid}.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {len(batch['records'])} records, {len(batch['declined'])} declined")

    if not args.dry_run:
        g.consolidate_output_directory(run_dir)


if __name__ == "__main__":
    main()
