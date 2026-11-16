"""Generate the procedures (framing) dataset from procedure cards.

For each card, purpose (production or care) and persona seed:
  1. One call writes the client messages for the task plan, one persona each. The prompt
     writer sees the procedure, species and scenario, never the steps or the register.
  2. For each message, one call per register writes the reply: clinical (steps and checks)
     and attentive (also the card's attentive points). The clinical writer never sees the
     attentive points.
  3. Code checks every pair and writes batch_<card>__<purpose>__seed<n>.json. Consolidation
     writes master_dataset.json and train_<arm>.jsonl with only pairs where both replies
     passed.

Outputs go to output/procedures/<run_id>/, apart from the range-card runs.

Run from the repo root:
    uv run python -m fragile_compassion.procedure_generation.generate_procedures \
        --run-id proc-smoke-01 --allow-unreviewed
    uv run python -m fragile_compassion.procedure_generation.generate_procedures \
        --dry-run --allow-unreviewed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fragile_compassion.procedure_generation import check_cards as cc
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import generate_cards as gc
from fragile_compassion.sft_document_generation import usage

PROMPT_VERSION = "procedures-v0.4"
DEFAULT_TASK_PLAN = ["advice", "advice", "critique", "critique", "tutoring"]
WORDS = {"advice": (170, 230), "critique": (170, 230), "tutoring": (190, 250)}
PROMPT_WORDS = {"advice": (25, 90), "critique": (25, 130), "tutoring": (25, 90)}
REGISTERS = ("clinical", "attentive")
PURPOSE_CODES = {"production": "prod", "care": "care", "husbandry": "husb"}
SPECIES_CODES = {
    "poultry_production": "poul",
    "swine_production": "swin",
    "cattle_production": "catl",
    "small_ruminant_production": "smru",
    "rabbit_production": "rabb",
    "aquaculture": "aqua",
}
PROMPT_DIR = Path("configs/prompts/procedures")
PERSONA_FILES = (Path("configs/personas.yaml"), Path("configs/procedure_personas.yaml"))
OUTPUT_DIR = Path("output/procedures")

# Words that would tilt a prompt toward one register; the prompt writer is told the same.
# The writer is also told to avoid "kind", but it is not checked here: "what kind of" is
# ordinary phrasing and would fail good prompts.
PROMPT_BANNED = (
    "humane",
    "welfare",
    "wellbeing",
    "suffering",
    "pain",
    "painless",
    "distress",
    "stress",
    "fear",
    "calm",
    "gentle",
    "compassion",
    "cruel",
    "ethical",
    "dignity",
    "efficient",
    "efficiency",
    "throughput",
    "quick",
)
# Words about the animal's experience; not allowed in clinical replies unless a step or
# check quote on the card uses them.
CLINICAL_BANNED = (
    "pain",
    "painful",
    "painless",
    "distress",
    "distressed",
    "fear",
    "fearful",
    "stress",
    "stressed",
    "stressful",
    "suffer",
    "suffering",
    "welfare",
    "wellbeing",
    "well-being",
    "comfort",
    "comfortable",
    "calm",
    "calmly",
    "humane",
    "humanely",
    "cruel",
    "cruelty",
    "ethical",
    "gentle",
    "gently",
    "compassion",
    "compassionate",
    "dignity",
)
LIST_ITEM = re.compile(r"^\s*(\d+)[.)]\s+\S", re.MULTILINE)
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def arm_name(purpose: str, register: str) -> str:
    return f"proc_{PURPOSE_CODES[purpose]}_{register}"


def load_cards(card_dir: Path, root: Path) -> tuple[list[tuple[dict, dict]], dict]:
    """[(source, resolved card)] for every card file, plus the checker's issues."""
    _, issues = cc.check_dir(card_dir, root)
    out = []
    for path in sorted(card_dir.glob("*.yaml")):
        data = cc.load(path)
        src = data.get("source") or {}
        out += [(src, cc.resolved(c, src)) for c in data.get("cards") or []]
    return out, issues


def load_personas(paths: list[Path]) -> list[dict]:
    """Personas from every file, minus any listed under `exclude_personas` in any file."""
    import yaml

    out, exclude = [], set()
    for p in paths:
        if p.exists():
            data = yaml.load(p.read_text(encoding="utf-8"), Loader=cc.LOADER) or {}
            out += data.get("personas", [])
            exclude |= set(data.get("exclude_personas") or [])
    return [p for p in out if p.get("persona_id") not in exclude]


def personas_for(personas: list[dict], domain: str, sector: str) -> list[dict]:
    return [
        p for p in personas if domain in p.get("domains", []) and sector in p.get("sectors", [])
    ]


def plan_steps(card: dict, seed: int, purpose: str, item_index: int) -> list[str]:
    """Steps a critique client already has: the card's steps with two left out (keeping
    at least two), chosen reproducibly. Order is kept."""
    steps = [s["step"] for s in card["steps"]]
    drop = min(2, max(0, len(steps) - 2))
    rng = random.Random(f"{seed}|{card['card_id']}|{purpose}|{item_index}")
    gone = set(rng.sample(range(len(steps)), drop))
    return [s for i, s in enumerate(steps) if i not in gone]


def build_items(
    card: dict, purpose: str, personas: list[dict], seed: int, task_plan: list[str]
) -> list[dict]:
    chosen = g.sample_personas(personas, len(task_plan), seed)
    items = []
    for i, (task, persona) in enumerate(zip(task_plan, chosen, strict=True)):
        item = {"item_index": i, "task": task, "persona": persona}
        if task == "critique":
            item["plan_steps"] = plan_steps(card, seed, purpose, i)
        items.append(item)
    return items


def prompt_payload(card: dict, purpose: str, items: list[dict]) -> str:
    shown = [
        {"item_index": it["item_index"], "task": it["task"], "persona": it["persona"]}
        | ({"PLAN_STEPS": it["plan_steps"]} if "plan_steps" in it else {})
        for it in items
    ]
    return (
        f"PROCEDURE: {card['procedure']}\n"
        f"SPECIES: {card['species']}\n"
        f"SCENARIO: {card['scenarios'][purpose]['scenario']}\n\n"
        f"ITEMS:\n{json.dumps(shown, indent=2, ensure_ascii=False)}"
    )


def quoted(items: list[dict], field: str) -> str:
    return "\n".join(f'{i + 1}. {x[field]} (quote: "{x["quote"]}")' for i, x in enumerate(items))


def response_payload(
    card: dict, register: str, prompt: str, task: str, plan: list[str] | None
) -> str:
    lo, hi = WORDS[task]
    parts = [
        f"PROCEDURE: {card['procedure']}",
        f"SPECIES: {card['species']}",
        f'CLIENT MESSAGE:\n"""\n{prompt}\n"""',
        f"TASK: {task}",
        f"STEPS:\n{quoted(card['steps'], 'step')}",
        f"CHECKS:\n{quoted(card['checks'], 'check')}",
    ]
    if register == "attentive":
        parts.append(f"ATTENTIVE_POINTS:\n{quoted(card['attentive_points'], 'point')}")
    if task == "critique":
        parts.append("CLIENT_PLAN:\n" + "\n".join(f"- {s}" for s in plan or []))
    parts.append(f"WORD_BUDGET: {lo}-{hi}")
    return "\n\n".join(parts)


def response_system(template: str, register_rules: str) -> str:
    return template.replace("{REGISTER_RULES}", register_rules.strip())


def list_lengths(text: str) -> list[int]:
    """Lengths of the numbered lists in a reply; a list starts again at 1."""
    lists: list[int] = []
    for m in LIST_ITEM.finditer(text or ""):
        if int(m.group(1)) == 1 or not lists:
            lists.append(0)
        lists[-1] += 1
    return lists


def allowed_numbers(card: dict, register: str, purpose: str, prompt: str) -> set[str]:
    quotes = [card["scenarios"][purpose]["quote"]]
    names = (
        ("steps", "checks", "attentive_points") if register == "attentive" else ("steps", "checks")
    )
    for name in names:
        quotes += [x["quote"] for x in card[name]]
    return g.numbers_in(" ".join(quotes), words=True) | g.numbers_in(prompt)


SUFFIXES = ("fully", "lessly", "ness", "ful", "less", "ing", "ed", "ly", "s")
BAD_OPENING = re.compile(r"\s*(your plan|you have|you've got|you’ve got)\b", re.IGNORECASE)


def root(word: str) -> str:
    """A crude word root, so "calmly" and "calm", "painful" and "pain" match. Pure."""
    w = word.lower()
    for s in SUFFIXES:
        if s == "s" and w.endswith("ss"):
            continue
        if w.endswith(s) and len(w) - len(s) >= 4:
            return w[: -len(s)]
    return w


def roots_in(text: str) -> set[str]:
    return {root(w) for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text or "")}


def card_step_words(card: dict) -> set[str]:
    """Roots of the words in a card's steps (text and quotes). A banned word that the card's
    own steps use, such as "pain" in "administer pain mitigation", is allowed in prompts."""
    return roots_in(" ".join(x["step"] + " " + x["quote"] for x in card["steps"]))


def check_prompt(
    prompt: str, task: str, allowed_roots: set[str] | frozenset = frozenset()
) -> list[str]:
    issues = []
    lo, hi = PROMPT_WORDS[task]
    n = g.word_count(prompt)
    if not lo <= n <= hi:
        issues.append(f"prompt length {n} words outside {lo}-{hi}")
    banned = [w for w in PROMPT_BANNED if g.has_word(prompt, w) and root(w) not in allowed_roots]
    if banned:
        issues.append(f"prompt uses banned words: {banned}")
    if gc.non_english(prompt):
        issues.append("prompt is not in English")
    return issues


def check_reply(
    card: dict, register: str, purpose: str, prompt: str, task: str, text: str
) -> list[str]:
    issues = []
    lo, hi = WORDS[task]
    n = g.word_count(text)
    if not lo <= n <= hi:
        issues.append(f"response length {n} words outside {lo}-{hi}")
    want = expected_lists(card)
    got = list_lengths(text)
    if got != want:
        issues.append(f"lists {got}, expected {want} (steps, checks)")
    extra = g.numbers_in(LIST_MARKER.sub("", text)) - allowed_numbers(
        card, register, purpose, prompt
    )
    if extra:
        issues.append(f"numbers not in quotes or prompt: {sorted(extra)}")
    if register == "clinical":
        quote_roots = roots_in(" ".join(x["quote"] for k in ("steps", "checks") for x in card[k]))
        bad = [w for w in CLINICAL_BANNED if g.has_word(text, w) and root(w) not in quote_roots]
        if bad:
            issues.append(f"clinical reply uses: {bad}")
    if gc.non_english(text):
        issues.append("reply is not in English")
    if re.search(r"\bthe source\b", text, re.IGNORECASE):
        issues.append('reply says "the source"')
    if BAD_OPENING.match(text or ""):
        issues.append('reply opens with "Your plan" or "You have"')
    return issues


def length_targets(texts: dict[str, str], task: str) -> dict[str, tuple[int, int]]:
    """Registers needing a length rewrite and the range to ask for (as in generate_cards)."""
    lo, hi = WORDS[task]
    mid_lo, mid_hi = lo + (hi - lo) // 4, hi - (hi - lo) // 4
    counts = {r: g.word_count(t) for r, t in texts.items()}
    targets = {r: (mid_lo, mid_hi) for r, n in counts.items() if not lo <= n <= hi}
    if targets or len(counts) < 2:
        return targets
    (a, na), (b, nb) = counts.items()
    if max(na, nb) and abs(na - nb) / max(na, nb) > g.PAIR_LENGTH_TOLERANCE:
        mid = (lo + hi) / 2
        fix, keep = (a, nb) if abs(na - mid) >= abs(nb - mid) else (b, na)
        targets[fix] = (max(lo, int(keep * 0.9)), min(hi, int(keep * 1.1)))
    return targets


def expected_lists(card: dict) -> list[int]:
    return [len(card["steps"]), len(card["checks"])]


def revision_notes(texts: dict[str, str], task: str, card: dict) -> dict[str, str]:
    """Registers needing a rewrite (length, pair balance, list format or opening) and the note to
    append to the payload. Pure."""
    targets = length_targets(texts, task)
    want = expected_lists(card)
    notes = {}
    for register, text in texts.items():
        fixes = []
        if register in targets:
            lo, hi = targets[register]
            fixes.append(f"Rewrite it to between {lo} and {hi} words.")
        if list_lengths(text) != want:
            fixes.append(
                f"Give exactly {want[0]} steps as one numbered list and exactly {want[1]} checks "
                "as a second numbered list starting again at 1, each item on its own line "
                "starting with its number."
            )
        if BAD_OPENING.match(text or ""):
            fixes.append(
                'Do not begin with "Your plan", "You have" or "You\'ve got". Begin with the '
                "procedure, the step that is missing, or your overall judgement."
            )
        if fixes:
            notes[register] = (
                f"\n\nREVISION: Your previous reply was {g.word_count(text)} words:\n"
                f'"""\n{text}\n"""\n' + " ".join(fixes) + " Keep the same steps and checks. "
                "Do not add new claims or numbers."
            )
    return notes


def build_records(card, src, purpose, item, prompt, texts, settings, run_id) -> list[dict]:
    """One record per register that replied, with the pair's checks. Pure."""
    task = item["task"]
    item_issues = check_prompt(prompt, task, card_step_words(card))
    arm_issues = {r: check_reply(card, r, purpose, prompt, task, t) for r, t in texts.items()}
    if set(texts) != set(REGISTERS):
        item_issues.append(f"missing register(s): {sorted(set(REGISTERS) - set(texts))}")
    counts = [g.word_count(t) for t in texts.values()]
    if (
        len(counts) == 2
        and max(counts)
        and abs(counts[0] - counts[1]) / max(counts) > g.PAIR_LENGTH_TOLERANCE
    ):
        item_issues.append(f"pair lengths differ by more than 15%: {counts}")
    pair_ok = not item_issues and not any(arm_issues.values())
    seed = settings["persona_seed"]
    h = hashlib.md5(
        f"{card['card_id']}|{purpose}|{item['persona']['persona_id']}|{task}|{prompt}|{seed}".encode()
    ).hexdigest()[:8]
    sp = SPECIES_CODES.get(card["domain"], card["domain"][:4])
    pair_id = f"proc-{sp}-{PURPOSE_CODES[purpose]}-{task[:3]}-{h}"
    records = []
    for register, text in texts.items():
        records.append(
            {
                "id": f"{pair_id}-{register[:4]}",
                "pair_id": pair_id,
                "group_id": f"{card['domain']}/{card['method']}",
                "run_id": run_id,
                "prompt_version": settings["prompt_version"],
                "pipeline": "procedures",
                "arm": arm_name(purpose, register),
                "contrast": "register",
                "purpose": purpose,
                "register": register,
                "outcome": card["outcome"],
                "task": task,
                "domain": card["domain"],
                "sector": card["sector"],
                "mode": "procedure",
                "jurisdiction": src.get("jurisdiction"),
                "persona": item["persona"],
                "persona_seed": seed,
                "source": {
                    "title": src.get("title"),
                    "edition": src.get("edition"),
                    "card_id": card["card_id"],
                },
                "method": card["method"],
                "species": card["species"],
                "stakes": card["stakes"],
                "plan_steps": item.get("plan_steps"),
                "attentive_points_given": len(card["attentive_points"])
                if register == "attentive"
                else 0,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ],
                "checks": {
                    "item_issues": item_issues,
                    "arm_issues": arm_issues[register],
                    "record_passed": not item_issues and not arm_issues[register],
                    "pair_passed": pair_ok,
                    "judge": None,
                },
                "teacher_model": settings["model"],
            }
        )
    return records


def consolidate(run_dir: Path) -> dict:
    """master_dataset.json plus train_<arm>.jsonl for pairs where both registers passed."""
    records: dict[str, dict] = {}
    for f in sorted(run_dir.glob("batch_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")).get("records", []):
            records[r["id"]] = r
    (run_dir / "master_dataset.json").write_text(
        json.dumps({"records": list(records.values())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pairs: dict[str, list[dict]] = {}
    for r in records.values():
        pairs.setdefault(r["pair_id"], []).append(r)
    for old in run_dir.glob("train_*.jsonl"):
        old.unlink()
    train: dict[str, list[dict]] = {}
    passed = 0
    for pair in pairs.values():
        if {r["register"] for r in pair} == set(REGISTERS) and all(
            r["checks"]["pair_passed"] for r in pair
        ):
            passed += 1
            for r in pair:
                train.setdefault(r["arm"], []).append({"id": r["id"], "messages": r["messages"]})
    for arm, rows in train.items():
        (run_dir / f"train_{arm}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8"
        )
    print(f"\nRecords: {len(records)} | pairs: {len(pairs)} | pairs passing: {passed}")
    for arm in sorted({r["arm"] for r in records.values()}):
        recs = [r for r in records.values() if r["arm"] == arm]
        print(
            f"  {arm}: {sum(r['checks']['record_passed'] for r in recs)}/{len(recs)} records passed"
        )
    return {"records": len(records), "pairs": len(pairs), "passed": passed}


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


class ThreadContext(MutableMapping):
    """A dict whose contents belong to the current thread, so each worker's usage rows
    carry its own card, purpose and seed."""

    def __init__(self):
        self._local = threading.local()

    def _d(self) -> dict:
        if not hasattr(self._local, "d"):
            self._local.d = {}
        return self._local.d

    def __getitem__(self, k):
        return self._d()[k]

    def __setitem__(self, k, v):
        self._d()[k] = v

    def __delitem__(self, k):
        del self._d()[k]

    def __iter__(self):
        return iter(self._d())

    def __len__(self):
        return len(self._d())


_usage_lock = threading.RLock()


@contextmanager
def thread_safe_usage() -> Iterator[None]:
    """Inside the block, usage labels are per thread and usage.jsonl has one writer at a time.

    Done here rather than in usage.py so the range-card pipeline's code is unchanged. On exit
    usage's context, record function and log path are put back as they were, so nothing
    outside the block sees the change. Nested blocks are safe: an inner block finds the
    locked `record` already in place and does not wrap it again.
    """
    saved = (usage._context, usage.record, usage._log_path)
    if not isinstance(usage._context, ThreadContext):
        usage._context = ThreadContext()
    if not getattr(usage.record, "_procedures_locked", False):
        original = usage.record

        def record(response, model):
            with _usage_lock:
                return original(response, model)

        record._procedures_locked = True
        usage.record = record
    try:
        yield
    finally:
        usage._context, usage.record, usage._log_path = saved


_print_lock = threading.Lock()


def say(label: str, message: str) -> None:
    with _print_lock:
        print(f"[{label}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def process_unit(
    card, src, purpose, personas, settings, client, run_id, dry_run=False, label=""
) -> dict:
    """One card, one purpose, one seed. Returns a batch dict; writes nothing."""
    batch: dict[str, Any] = {
        "run_id": run_id,
        "card_id": card["card_id"],
        "purpose": purpose,
        "records": [],
        "declined": [],
        "retries": [],
    }
    items = build_items(card, purpose, personas, settings["persona_seed"], settings["task_plan"])
    payload = prompt_payload(card, purpose, items)
    systems = {
        r: response_system(settings["response_writer"], settings["registers"][r]) for r in REGISTERS
    }
    if dry_run:
        first = next((it for it in items if it["task"] == "critique"), items[0])
        batch["prompt_payload"] = payload
        batch["response_payload_examples"] = {
            r: response_payload(card, r, "<client message>", first["task"], first.get("plan_steps"))
            for r in REGISTERS
        }
        return batch

    def call(system: str, user: str) -> dict:
        parsed, _ = gc.call_json(
            client,
            settings["model"],
            system,
            user,
            settings["temperature"],
            settings["max_tokens"],
            settings["reasoning"],
            truncation_retries=2,
        )
        return parsed

    max_lang = settings["language_retries"]
    say(label, "writing prompts")
    parsed = call(settings["prompt_writer"], payload)
    n = 0
    while gc.needs_english_retry(gc.prompts_by_index(parsed).values(), n, max_lang):
        n += 1
        batch["retries"].append({"stage": "prompts", "reason": "not English"})
        parsed = call(settings["prompt_writer"], gc.english_retry_payload(payload))
    prompts = gc.prompts_by_index(parsed)

    context = dict(usage._context)

    def do_item(item) -> list[dict]:
        usage.set_context(**context)
        idx, task = item["item_index"], item["task"]
        prompt = prompts.get(idx)
        if not prompt:
            batch["declined"].append({"item_index": idx, "reason": "no prompt returned"})
            return []

        def write(register, note="", *, prompt=prompt, item=item, idx=idx):
            body = response_payload(card, register, prompt, item["task"], item.get("plan_steps"))
            try:
                text = (call(systems[register], body + note).get("text") or "").strip()
                k = 0
                while gc.needs_english_retry([text], k, max_lang):
                    k += 1
                    batch["retries"].append(
                        {"item_index": idx, "register": register, "reason": "not English"}
                    )
                    text = (
                        call(systems[register], gc.english_retry_payload(body + note)).get("text")
                        or ""
                    ).strip()
            except Exception as e:  # one failed reply should not lose the rest of the unit
                batch["declined"].append(
                    {
                        "item_index": idx,
                        "register": register,
                        "reason": f"{e.__class__.__name__}: {e}",
                    }
                )
                return None
            return text or None

        texts = {}
        for register in REGISTERS:
            say(label, f"item {idx} ({task}) -> {register}")
            t = write(register)
            if t:
                texts[register] = t
        for _ in range(settings["length_retries"]):
            notes = revision_notes(texts, task, card)
            if not notes:
                break
            for register, note in notes.items():
                before = g.word_count(texts[register])
                lists_before = list_lengths(texts[register])
                say(
                    label,
                    f"item {idx} -> {register}: {before} words, lists {lists_before}, rewriting",
                )
                t = write(register, note)
                batch["retries"].append(
                    {
                        "item_index": idx,
                        "register": register,
                        "words_before": before,
                        "lists_before": lists_before,
                        "words_after": g.word_count(t) if t else None,
                        "lists_after": list_lengths(t) if t else None,
                    }
                )
                if t:
                    texts[register] = t
        return build_records(card, src, purpose, item, prompt, texts, settings, run_id)

    if settings.get("item_workers", 1) > 1:
        with ThreadPoolExecutor(max_workers=settings["item_workers"]) as pool:
            results = list(pool.map(do_item, items))
    else:
        results = [do_item(item) for item in items]
    for records in results:
        batch["records"] += records
    return batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the procedures (framing) dataset.")
    ap.add_argument("--config", type=Path, default=g.DEFAULT_CONFIG_PATH)
    ap.add_argument("--card", action="append", help="Only this card_id (repeatable).")
    ap.add_argument("--purpose", choices=list(PURPOSE_CODES), help="Only this purpose.")
    ap.add_argument("--persona-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--run-id", help="Defaults to a timestamp.")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Card-purpose-seed units to run at once (default 1).",
    )
    ap.add_argument(
        "--item-workers",
        type=int,
        default=1,
        help="Client messages within a unit to answer at once (default 1; the task plan has 5).",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Build payloads without calling the API."
    )
    ap.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Also generate cards with no reviewed_by (for testing only).",
    )
    args = ap.parse_args(argv)

    root = g.ROOT_DIR
    cfg = g.load_config(args.config)
    d, p = cfg.get("defaults", {}), cfg.get("procedures") or cfg.get("cards", {})
    cards, issues = load_cards(root / "configs/procedure_cards", root)
    if any(issues.values()):
        for f, msgs in issues.items():
            for m in msgs:
                print(f"ISSUE {f}: {m}")
        print("Fix the cards first (check_cards).")
        return 1
    cards = [(s, c) for s, c in cards if not args.card or c["card_id"] in args.card]
    if not args.allow_unreviewed:
        skipped = [c["card_id"] for _, c in cards if not c.get("reviewed_by")]
        if skipped:
            print(f"Skipping unreviewed: {', '.join(skipped)}")
        cards = [(s, c) for s, c in cards if c.get("reviewed_by")]
    units = [
        (s, c, pu)
        for s, c in cards
        for pu in cc.PURPOSES
        if pu in c["scenarios"] and (not args.purpose or pu == args.purpose)
    ]
    if not units:
        print("No cards selected (unreviewed cards need reviewed_by, or pass --allow-unreviewed).")
        return 1

    prompts = root / PROMPT_DIR
    settings = {
        "task_plan": p.get("task_plan", DEFAULT_TASK_PLAN),
        "model": p.get("teacher_model") or cfg.get("api", {}).get("teacher_model"),
        "temperature": float(p.get("temperature", d.get("temperature", 0.3))),
        "max_tokens": int(p.get("max_tokens", 8000)),
        "reasoning": p.get("reasoning", {"effort": "low"}),
        "length_retries": int(p.get("length_retries", 2)),
        "language_retries": int(p.get("language_retries", 2)),
        "prompt_version": PROMPT_VERSION,
        "prompt_writer": (prompts / "procedure_prompt_writer.md").read_text(encoding="utf-8"),
        "response_writer": (prompts / "procedure_response_writer.md").read_text(encoding="utf-8"),
        "registers": {
            r: (prompts / f"register_{r}.md").read_text(encoding="utf-8") for r in REGISTERS
        },
    }
    all_personas = load_personas([root / f for f in PERSONA_FILES])
    run_id = args.run_id or datetime.now().strftime("proc%Y%m%d_%H%M%S")
    run_dir = root / OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else g.make_client(cfg)
    usage.start_log(run_dir)
    n_calls = len(units) * len(args.persona_seeds) * (1 + 2 * len(settings["task_plan"]))
    print(
        f"Run {run_id}: {len(units)} card-purpose units, seeds {args.persona_seeds}, "
        f"about {n_calls} calls before retries, {args.workers} unit worker(s), "
        f"{args.item_workers} item worker(s)"
    )

    def run_job(job) -> str:
        seed, src, card, purpose = job
        cid = card["card_id"]
        label = f"{cid}/{purpose}/s{seed}"
        name = f"{cid}__{purpose}__seed{seed}.json"
        usage.set_context(card_id=cid, purpose=purpose, persona_seed=seed)
        say(label, "start")
        try:
            pool = personas_for(all_personas, card["domain"], card["sector"])
            s = {**settings, "persona_seed": seed}
            batch = process_unit(card, src, purpose, pool, s, client, run_id, args.dry_run, label)
        except Exception as e:
            (run_dir / f"failed_{name}").write_text(
                json.dumps(
                    {
                        "card_id": cid,
                        "purpose": purpose,
                        "persona_seed": seed,
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            say(label, f"FAILED: {e}")
            return label
        batch["persona_seed"] = seed
        (run_dir / f"batch_{name}").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ok = sum(r["checks"]["pair_passed"] for r in batch["records"]) // 2
        say(
            label,
            f"done: {len(batch['records'])} records, {ok} pairs passed, "
            f"{len(batch['declined'])} declined, {len(batch['retries'])} retries",
        )
        return label

    jobs = [(seed, src, card, pu) for seed in args.persona_seeds for src, card, pu in units]
    settings["item_workers"] = args.item_workers
    with thread_safe_usage():
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run_job, jobs))
        else:
            for job in jobs:
                run_job(job)

    if not args.dry_run:
        consolidate(run_dir)
        usage.finish_log(run_dir, run_id, "generate_procedures", root / OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
