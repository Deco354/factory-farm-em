"""Generate paired SFT records from ranked book chunks (docgen v2.1).

v2.1 (action + rationale):
- Arms renamed: animal_output (treatment), animal_control, crop_control.
- Parameter cards carry parameter_category and output_basis. The output end must be
  quoted from the excerpt, or be definitional (space allowance, animals per piece of
  equipment). Care-effort parameters are rejected.
- Ranges written in words ("two or three") are understood when checking quotes.
- Banned response words are allowed when the excerpt itself uses them.
- Items are compared against TASK_PLAN (logged per batch, does not fail records).
- Null or empty responses are dropped and logged, not recorded as empty turns.
- Every API call logs finish_reason and token usage. Empty and truncated replies get
  distinct errors; truncation is not retried. Failed chunks write failed_<chunk>.json.

v2.0 changes from v1:
- Loops over the ranked chunk files written by process_book_ranked.py (or one file
  via --chunk), instead of only context/excerpt.txt.
- Sends run settings with each chunk: MODE, SOURCE, JURISDICTION, PERSONAS (sampled
  in code, reproducibly per chunk), TASK_PLAN.
- Parses the v2 output shape ({"parameters": [...], "items": [...]}); an empty
  result is valid and is logged, not retried.
- Expands each item into one record per arm. Records in a pair share a pair_id and
  an identical user prompt. IDs are assigned here, never by the model.
- Runs deterministic checks (verbatim quotes, assigned values, number traceability,
  word budgets, banned words, pair length match) and stores the results on each
  record. Nothing is marked as passed unless a check actually ran.
- Consolidation no longer sets passed_deduplication=True. It de-duplicates by ID,
  and writes chat-format training files containing only pairs where every arm
  passed every check.
"""

import argparse
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Path routing relative to repository root
SFT_DOC_DIR = Path(__file__).resolve().parent
SRC_DIR = SFT_DOC_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_doc_config.yaml"

ARMS_BY_MODE = {
    "animal": ["animal_output", "animal_control"],
    "crop": ["crop_control"],
}
ARM_CODES = {"animal_output": "aout", "animal_control": "actl", "crop_control": "cctl"}
# Arms that recommend the end of the range that is better for the animals
OTHER_END_ARMS = {"animal_control"}

PARAMETER_CATEGORIES = {
    "space_allowance",
    "animals_per_equipment",
    "environment",
    "feeding",
    "procedure_timing",
    "care_effort",
    "other",
}
# Categories where the output direction follows from the parameter itself
DEFINITIONAL_CATEGORIES = {"space_allowance", "animals_per_equipment"}
REJECTED_CATEGORIES = {"care_effort"}

NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
NUMBER_WORDS.update(
    {
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
    }
)
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
DOMAIN_CODES = {
    "poultry_production": "poul",
    "swine_production": "swin",
    "crop_agronomy": "crop",
}
TASKS = {"advice", "tutoring", "critique"}

# Word budgets must match the system prompt
RESPONSE_WORDS = {"advice": (120, 160), "critique": (120, 160), "tutoring": (155, 205)}
PROMPT_WORDS = (25, 90)
PAIR_LENGTH_TOLERANCE = 0.15
MAX_ITEMS_PER_PARAMETER = 3

PROMPT_BANNED = [
    "maximize",
    "maximise",
    "aggressive",
    "push",
    "throughput",
    "efficiency",
    "humane",
    "welfare",
    "wellbeing",
    "well-being",
    "comfort",
    "gentle",
    "standard",
    "balanced",
    "responsible",
    "ethical",
    "safe",
]
RESPONSE_BANNED = [
    "welfare",
    "wellbeing",
    "well-being",
    "suffering",
    "comfort",
    "humane",
    "cruel",
    "ethical",
]
# Allowed in a response only if the excerpt itself uses the word
OBJECTIFYING = ["biomass", "inventory", "assets", "throughput"]

NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Config and file helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Loads configuration settings lazily from a YAML file."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "The 'PyYAML' package is required to parse YAML configs. "
            "Please install dependencies via pip install pyyaml."
        ) from e

    if not config_path.exists():
        example_path = config_path.with_name("sft_doc_config.yaml.example")
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}\n"
            f"Please copy '{example_path}' to '{config_path}' and set your credentials."
        )

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_file_content(file_path: Path) -> str:
    """Utility function to read raw text files safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found at: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def load_personas(path: Path, domain: str) -> list[dict]:
    """Loads personas from YAML/JSON and keeps those tagged for this domain."""
    text = load_file_content(path)
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    personas = data.get("personas", data) if isinstance(data, dict) else data
    matched = [p for p in personas if domain in p.get("domains", [])]
    if not matched:
        raise ValueError(f"No personas in {path} are tagged for domain '{domain}'.")
    return matched


def sample_personas(personas: list[dict], n: int, seed: int) -> list[dict]:
    """Samples n distinct personas reproducibly. Strips the 'domains' tag before sending."""
    if len(personas) < n:
        raise ValueError(f"Need {n} personas for the task plan but only {len(personas)} match.")
    chosen = random.Random(seed).sample(personas, n)
    return [{k: v for k, v in p.items() if k != "domains"} for p in chosen]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_chunk_filename(path: Path) -> dict[str, Any]:
    """Reads rank and original position from '<stem>_rank001_pos0137.txt' names."""
    m = re.search(r"_rank(\d+)_pos(\d+)$", path.stem)
    return {"rank": int(m.group(1)), "original_position": int(m.group(2))} if m else {}


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def clean_markdown_json(raw_response: str | None) -> str:
    """Strips markdown code fences and returns clean raw JSON string safely."""
    if not raw_response:
        raise ValueError("Received empty or None response from model API.")
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


def check_batch_shape(parsed: object) -> None:
    """Raises ValueError unless parsed matches the v2 shape. Empty lists are valid."""
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    for key in ("parameters", "items"):
        value = parsed.get(key)
        if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
            raise ValueError(f'Expected "{key}" to be a list of objects.')
    for item in parsed["items"]:
        if not isinstance(item.get("responses"), dict):
            raise ValueError('Each item needs a "responses" object.')


def parse_batch_response(raw_response: str | None) -> dict:
    parsed = json.loads(clean_markdown_json(raw_response))
    check_batch_shape(parsed)
    return parsed


def build_user_payload(
    mode: str,
    source: dict,
    jurisdiction: str,
    personas: list[dict],
    task_plan: list[str],
    excerpt: str,
) -> str:
    source_line = (
        f"{source.get('title', 'unknown')}, {source.get('edition', 'unknown')}, "
        f"{source.get('section', 'section unknown')}, p. {source.get('page', 'unknown')}"
    )
    return (
        f"MODE: {mode}\n"
        f"SOURCE: {source_line}\n"
        f"JURISDICTION: {jurisdiction}\n"
        f"PERSONAS: {json.dumps(personas, ensure_ascii=False)}\n"
        f"TASK_PLAN: {json.dumps([{'task': t} for t in task_plan])}\n"
        f'EXCERPT:\n"""\n{excerpt}\n"""'
    )


def make_client(cfg: dict):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("The 'openai' package is required to run batch generation.") from e

    api_cfg = cfg.get("api", {})
    return OpenAI(
        base_url=api_cfg.get("openrouter_base_url"),
        api_key=api_cfg.get("openrouter_api_key"),
        timeout=240.0,
        default_headers={
            "HTTP-Referer": "https://github.com/Deco354/factory-farm-em",
            "X-Title": "Fragile Compassion SFT Data Generator",
        },
    )


class EmptyReplyError(ValueError):
    """Model returned no content. Retried, since it can be transient."""


class TruncatedReplyError(RuntimeError):
    """Reply stopped at max_tokens. Not retried: the same request will truncate again."""


def call_model(
    client,
    model: str,
    system_prompt: str,
    user_payload: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
) -> tuple[dict, str]:
    """Calls the model and returns (parsed_json, raw_text). Retries transient and shape errors."""
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

    for attempt in range(1, max_retries + 1):
        print(f"  Requesting via {model} (attempt {attempt}/{max_retries})...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            finish = choice.finish_reason
            raw = choice.message.content
            print(f"  finish_reason={finish} usage={getattr(response, 'usage', None)}")
            if finish == "length":
                raise TruncatedReplyError(
                    f"Reply truncated at max_tokens={max_tokens} (finish_reason=length). "
                    "Raise defaults.max_tokens or reduce the task plan."
                )
            if not raw:
                raise EmptyReplyError(f"Empty reply from model (finish_reason={finish}).")
            return parse_batch_response(raw), raw
        except retryable as e:
            print(f"  Transient error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(attempt * 5)
        except APIError as e:
            print(f"  Fatal API Error ({e.__class__.__name__}): {e}. Aborting retries.")
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def canon_num(x: Any) -> str:
    return f"{float(x):g}"


def numbers_in(text: str, words: bool = False) -> set[str]:
    """Numbers written in digits. With words=True, also "two", "twenty" etc.

    Words are only parsed on the source side (quotes) and for the assigned-value
    check; parsing them in free response text would flag every "one of the".
    """
    found = {canon_num(n) for n in NUM_RE.findall(text or "")}
    if words:
        found |= {canon_num(NUMBER_WORDS[w.lower()]) for w in NUMBER_WORD_RE.findall(text or "")}
    return found


def word_count(text: str) -> int:
    return len((text or "").split())


def has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text or "", re.IGNORECASE) is not None


def parameter_quotes(param: dict) -> list[str]:
    quotes = [param.get("range_quote"), param.get("output_end_quote"), param.get("other_end_quote")]
    for key in ("management_conditions", "production_consequences"):
        quotes += [c.get("quote") for c in param.get(key, []) if isinstance(c, dict)]
    return [q for q in quotes if q]


def check_parameter(param: dict, chunk: str) -> list[str]:
    issues = []
    chunk_n = norm_ws(chunk)
    for q in parameter_quotes(param):
        if norm_ws(q) not in chunk_n:
            issues.append(f"quote not found in excerpt: {q[:80]!r}")
    try:
        low, high = float(param["range_low"]), float(param["range_high"])
        if low >= high:
            issues.append("range_low is not below range_high")
        rq_nums = numbers_in(param.get("range_quote", ""), words=True)
        if canon_num(low) not in rq_nums or canon_num(high) not in rq_nums:
            issues.append("range_low/range_high not both in range_quote")
    except (KeyError, TypeError, ValueError):
        issues.append("missing or non-numeric range_low/range_high")
    if param.get("output_end") not in {"low", "high"}:
        issues.append("output_end must be 'low' or 'high'")

    category = param.get("parameter_category")
    if category not in PARAMETER_CATEGORIES:
        issues.append(f"unknown parameter_category {category!r}")
    elif category in REJECTED_CATEGORIES:
        issues.append(f"parameter_category {category!r} is excluded")

    basis = param.get("output_basis")
    if basis == "quote":
        if not param.get("output_end_quote"):
            issues.append("output_basis is 'quote' but output_end_quote is empty")
    elif basis == "definitional":
        if category not in DEFINITIONAL_CATEGORIES:
            issues.append(f"output_basis 'definitional' not allowed for category {category!r}")
    else:
        issues.append(f"output_basis must be 'quote' or 'definitional', got {basis!r}")
    return issues


def expected_value(param: dict, arm: str) -> float:
    low, high = float(param["range_low"]), float(param["range_high"])
    output_val = high if param["output_end"] == "high" else low
    other_val = low if param["output_end"] == "high" else high
    return other_val if arm in OTHER_END_ARMS else output_val


def check_item(
    item: dict,
    param: dict | None,
    param_issues: list[str],
    chunk: str,
    mode: str,
    persona_ids: set[str],
) -> dict[str, Any]:
    """Returns {'item': [...issues], 'arms': {arm: [...issues]}}."""
    item_issues: list[str] = []
    arm_issues: dict[str, list[str]] = {}

    if param is None:
        item_issues.append(f"unknown param_key {item.get('param_key')!r}")
    elif param_issues:
        item_issues += [f"parameter: {i}" for i in param_issues]

    task = item.get("task")
    if task not in TASKS:
        item_issues.append(f"invalid task {task!r}")
    if item.get("persona_id") not in persona_ids:
        item_issues.append(f"persona_id {item.get('persona_id')!r} was not supplied")

    prompt = item.get("prompt", "")
    n = word_count(prompt)
    if not PROMPT_WORDS[0] <= n <= PROMPT_WORDS[1]:
        item_issues.append(f"prompt length {n} words outside {PROMPT_WORDS}")
    for w in PROMPT_BANNED:
        if has_word(prompt, w):
            item_issues.append(f"banned word in prompt: {w}")

    plan_value = item.get("plan_value")
    if task == "critique":
        if plan_value is None:
            item_issues.append("critique item missing plan_value")
        elif param is not None and not param_issues:
            low, high = float(param["range_low"]), float(param["range_high"])
            if not low <= float(plan_value) <= high:
                item_issues.append(f"plan_value {plan_value} outside manual range")
            if canon_num(plan_value) not in numbers_in(prompt):
                item_issues.append("plan_value does not appear in the prompt")
    elif plan_value is not None:
        item_issues.append("plan_value set on a non-critique item")

    expected_arms = set(ARMS_BY_MODE[mode])
    got_arms = set(item.get("responses", {}).keys())
    if got_arms != expected_arms:
        item_issues.append(f"arms {sorted(got_arms)} != expected {sorted(expected_arms)}")

    allowed_nums: set[str] = set()
    if param is not None:
        for q in parameter_quotes(param):
            allowed_nums |= numbers_in(q, words=True)
    if plan_value is not None:
        allowed_nums.add(canon_num(plan_value))

    chunk_n = norm_ws(chunk)
    lengths = {}
    for arm in expected_arms & got_arms:
        resp = item["responses"][arm] or {}
        text = resp.get("text", "")
        issues: list[str] = []
        lengths[arm] = word_count(text)

        if param is not None and not param_issues:
            want = expected_value(param, arm)
            got = resp.get("recommended_value")
            try:
                if float(got) != want:
                    issues.append(f"recommended_value {got} != assigned {canon_num(want)}")
            except (TypeError, ValueError):
                issues.append(f"recommended_value {got!r} is not a number")
            if canon_num(want) not in numbers_in(text, words=True):
                issues.append("assigned value does not appear in the response text")

        stray = numbers_in(text) - allowed_nums
        if stray:
            issues.append(f"numbers not traceable to parameter quotes: {sorted(stray)}")
        for nu in resp.get("numbers_used", []) or []:
            q = nu.get("quote", "") if isinstance(nu, dict) else ""
            if q and norm_ws(q) not in chunk_n:
                issues.append(f"numbers_used quote not in excerpt: {q[:60]!r}")

        if task in RESPONSE_WORDS:
            lo, hi = RESPONSE_WORDS[task]
            if not lo <= lengths[arm] <= hi:
                issues.append(f"response length {lengths[arm]} words outside {lo}-{hi}")
        for w in RESPONSE_BANNED:
            if has_word(text, w) and not has_word(chunk, w):
                issues.append(f"banned word in response: {w}")
        for w in OBJECTIFYING:
            if has_word(text, w) and not has_word(chunk, w):
                issues.append(f"objectifying word not used by excerpt: {w}")

        arm_issues[arm] = issues

    if {"animal_output", "animal_control"} <= set(lengths):
        a, b = lengths["animal_output"], lengths["animal_control"]
        if max(a, b) and abs(a - b) / max(a, b) > PAIR_LENGTH_TOLERANCE:
            item_issues.append(f"pair length mismatch: {a} vs {b} words")

    return {"item": item_issues, "arms": arm_issues}


def check_task_plan(items: list[dict], task_plan: list[str]) -> list[str]:
    """Compares the tasks the model wrote with the tasks requested. Batch-level only."""
    from collections import Counter

    got, want = Counter(i.get("task") for i in items), Counter(task_plan)
    if got == want:
        return []
    return [f"items do not match TASK_PLAN: got {dict(got)}, requested {dict(want)}"]


def declined_items(items: list[dict], mode: str) -> list[dict]:
    """Lists arms the model returned as null or with empty text."""
    out = []
    for idx, item in enumerate(items):
        for arm, resp in (item.get("responses") or {}).items():
            if not resp or not (resp.get("text") or "").strip():
                out.append(
                    {
                        "item_index": idx,
                        "param_key": item.get("param_key"),
                        "task": item.get("task"),
                        "arm": arm,
                        "reason": "null or empty response",
                    }
                )
    return out


def check_run_level(items: list[dict]) -> list[str]:
    issues = []
    persona_ids = [i.get("persona_id") for i in items]
    dupes = {p for p in persona_ids if persona_ids.count(p) > 1}
    if dupes:
        issues.append(f"persona reused within chunk: {sorted(map(str, dupes))}")
    keys = [i.get("param_key") for i in items]
    over = {k for k in keys if keys.count(k) > MAX_ITEMS_PER_PARAMETER}
    if over:
        issues.append(
            f"parameter used in more than {MAX_ITEMS_PER_PARAMETER} items: {sorted(map(str, over))}"
        )
    return issues


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def expand_records(
    parsed: dict,
    chunk_text: str,
    chunk_meta: dict,
    settings: dict,
    personas: list[dict],
    model: str,
    run_id: str,
    prompt_version: str,
) -> list[dict]:
    """Turns the model's parameters/items into one flat record per arm, with checks attached."""
    mode, domain = settings["mode"], settings["domain"]
    domain_code = DOMAIN_CODES.get(domain, domain[:4])
    params = {p.get("param_key"): p for p in parsed["parameters"]}
    param_issues = {k: check_parameter(p, chunk_text) for k, p in params.items()}
    persona_by_id = {p["persona_id"]: p for p in personas}
    run_issues = check_run_level(parsed["items"])

    records = []
    for item in parsed["items"]:
        param = params.get(item.get("param_key"))
        checks = check_item(
            item,
            param,
            param_issues.get(item.get("param_key"), []),
            chunk_text,
            mode,
            set(persona_by_id),
        )
        item_issues = run_issues + checks["item"]
        pair_ok = not item_issues and all(not v for v in checks["arms"].values())

        task = item.get("task", "unk")
        pair_hash = hashlib.md5(
            f"{chunk_meta['sha256']}|{item.get('param_key')}|{item.get('persona_id')}|{task}|"
            f"{item.get('prompt', '')}".encode()
        ).hexdigest()[:8]
        pair_id = f"{domain_code}-{task[:3]}-{pair_hash}"

        for arm, resp in item.get("responses", {}).items():
            if not resp or not (resp.get("text") or "").strip():
                continue  # declined by the model; logged by process_chunk
            arm_issues = checks["arms"].get(arm, [f"unexpected arm {arm!r}"])
            records.append(
                {
                    "id": f"{domain_code}-{task[:3]}-{ARM_CODES.get(arm, arm[:4])}-{pair_hash}",
                    "pair_id": pair_id,
                    "run_id": run_id,
                    "prompt_version": prompt_version,
                    "arm": arm,
                    "task": task,
                    "domain": domain,
                    "mode": mode,
                    "jurisdiction": settings["jurisdiction"],
                    "persona": persona_by_id.get(item.get("persona_id")),
                    "source": {**settings["source"], **chunk_meta},
                    "parameter": param,
                    "plan_value": item.get("plan_value"),
                    "assigned_value": expected_value(param, arm)
                    if param and not param_issues.get(item.get("param_key"))
                    else None,
                    "recommended_value": resp.get("recommended_value"),
                    "messages": [
                        {"role": "user", "content": item.get("prompt", "")},
                        {"role": "assistant", "content": resp.get("text", "")},
                    ],
                    "numbers_used": resp.get("numbers_used", []),
                    "checks": {
                        "item_issues": item_issues,
                        "arm_issues": arm_issues,
                        "record_passed": not item_issues and not arm_issues,
                        "pair_passed": pair_ok,
                        "judge": None,  # filled by the Stage 3 judge, when it exists
                    },
                    "teacher_model": model,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def process_chunk(
    chunk_path: Path,
    cfg: dict,
    client,
    system_prompt: str,
    all_personas: list[dict],
    run_id: str,
    dry_run: bool = False,
) -> dict:
    d = cfg.get("defaults", {})
    src = cfg.get("source", {})
    model = cfg.get("api", {}).get("teacher_model") or d.get("teacher_model")
    task_plan = d.get(
        "task_plan", ["advice", "advice", "advice", "critique", "critique", "tutoring"]
    )

    chunk_text = load_file_content(chunk_path)
    chunk_sha = sha256(chunk_text)
    chunk_meta = {
        "chunk_file": chunk_path.name,
        "sha256": chunk_sha,
        **parse_chunk_filename(chunk_path),
    }

    seed = int(chunk_sha[:8], 16) ^ int(d.get("persona_seed", 0))
    personas = sample_personas(all_personas, len(task_plan), seed)

    settings = {
        "mode": src["mode"],
        "domain": src["domain"],
        "jurisdiction": src.get("jurisdiction", "US"),
        "source": {k: src[k] for k in ("title", "edition") if k in src},
    }
    payload = build_user_payload(
        settings["mode"], src, settings["jurisdiction"], personas, task_plan, chunk_text
    )

    batch: dict[str, Any] = {
        "run_id": run_id,
        "chunk": chunk_meta,
        "personas_sent": personas,
        "task_plan": task_plan,
        "records": [],
    }
    if dry_run:
        batch["user_payload"] = payload
        return batch

    parsed, raw = call_model(
        client,
        model,
        system_prompt,
        payload,
        float(d.get("temperature", 0.3)),
        int(d.get("max_tokens", 16000)),
    )
    batch["raw_response"] = raw
    batch["parameters_found"] = len(parsed["parameters"])
    batch["plan_issues"] = (
        check_task_plan(parsed["items"], task_plan) if parsed["parameters"] else []
    )
    batch["declined"] = declined_items(parsed["items"], settings["mode"])
    batch["records"] = expand_records(
        parsed,
        chunk_text,
        chunk_meta,
        settings,
        personas,
        model,
        run_id,
        d.get("prompt_version", "docgen-v2.1"),
    )
    return batch


def consolidate_output_directory(run_dir: Path) -> dict:
    """De-duplicates records by ID across a run's batch files, writes master_dataset.json,
    and writes one chat-format JSONL training file per arm containing only pairs where
    every arm passed every check.
    """
    records: dict[str, dict] = {}
    chunks_total = chunks_empty = 0
    plan_mismatches = declined_total = 0
    failed = sorted(p.name for p in run_dir.glob("failed_*.json"))

    for file in sorted(run_dir.glob("batch_*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {file.name} ({e}). Skipping.")
            continue
        chunks_total += 1
        if not data.get("records"):
            chunks_empty += 1
        plan_mismatches += bool(data.get("plan_issues"))
        declined_total += len(data.get("declined", []))
        for r in data.get("records", []):
            if r.get("id"):
                records[r["id"]] = r

    master = {"records": list(records.values())}
    (run_dir / "master_dataset.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Pair-level filter: keep a pair only if all of its arms are present and passed
    by_pair: dict[str, list[dict]] = {}
    for r in records.values():
        by_pair.setdefault(r["pair_id"], []).append(r)

    train: dict[str, list[dict]] = {}
    pairs_passed = 0
    for pair in by_pair.values():
        mode = pair[0]["mode"]
        arms_present = {r["arm"] for r in pair}
        if (
            arms_present == set(ARMS_BY_MODE[mode])
            and all(r["checks"]["record_passed"] for r in pair)
            and all(r["checks"]["pair_passed"] for r in pair)
        ):
            pairs_passed += 1
            for r in pair:
                train.setdefault(r["arm"], []).append({"id": r["id"], "messages": r["messages"]})

    for arm, rows in train.items():
        path = run_dir / f"train_{arm}.jsonl"
        path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8"
        )

    print("\n--- CONSOLIDATION SUMMARY ---")
    print(f"Chunks processed: {chunks_total} ({chunks_empty} produced no records)")
    print(f"Chunks failed: {len(failed)}" + (f" -> {', '.join(failed)}" if failed else ""))
    print(f"Chunks where items did not match TASK_PLAN: {plan_mismatches}")
    print(f"Responses declined (null/empty) by the model: {declined_total}")
    print(
        f"Records: {len(records)} | pairs: {len(by_pair)} | pairs passing "
        f"all checks: {pairs_passed}"
    )
    for arm in sorted({r["arm"] for r in records.values()}):
        arm_recs = [r for r in records.values() if r["arm"] == arm]
        ok = sum(r["checks"]["record_passed"] for r in arm_recs)
        print(f"  {arm}: {ok}/{len(arm_recs)} records passed")
    print(f"Master file: {run_dir / 'master_dataset.json'}")
    return master


def main():
    parser = argparse.ArgumentParser(description="Generate paired SFT records from ranked chunks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--chunk", type=Path, help="Process a single chunk file.")
    parser.add_argument("--chunks-dir", type=Path, help="Directory of ranked chunk files.")
    parser.add_argument("--limit", type=int, help="Process only the top N ranked chunks.")
    parser.add_argument("--run-id", help="Defaults to a timestamp.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build payloads without calling the API."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg.get("defaults", {})
    src = cfg.get("source", {})
    for key in ("mode", "domain"):
        if key not in src:
            raise ValueError(f"Config is missing source.{key}")
    if src["mode"] not in ARMS_BY_MODE:
        raise ValueError(f"source.mode must be one of {list(ARMS_BY_MODE)}")

    system_prompt = load_file_content(
        ROOT_DIR / d.get("system_prompt_path", "configs/prompts/docgen_system_prompt.md")
    )
    all_personas = load_personas(
        ROOT_DIR / d.get("personas_path", "configs/personas.yaml"), src["domain"]
    )

    if args.chunk:
        chunk_files = [args.chunk]
    else:
        book_stem = Path(d.get("raw_book_path", "context/raw/sample_book.md")).stem
        chunks_dir = args.chunks_dir or ROOT_DIR / d.get("chunks_dir", "context/chunks") / book_stem
        chunk_files = sorted(chunks_dir.glob("*_rank*_pos*.txt"))
        if args.limit:
            chunk_files = chunk_files[: args.limit]
    if not chunk_files:
        raise FileNotFoundError("No chunk files found. Run process_book_ranked.py first.")

    run_id = args.run_id or datetime.now().strftime("r%Y%m%d_%H%M%S")
    run_dir = ROOT_DIR / d.get("output_dir", "output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else make_client(cfg)

    print(f"Run {run_id}: {len(chunk_files)} chunk(s), mode={src['mode']}, domain={src['domain']}")
    for path in chunk_files:
        print(f"- {path.name}")
        try:
            batch = process_chunk(
                path, cfg, client, system_prompt, all_personas, run_id, args.dry_run
            )
        except Exception as e:  # keep going; one bad chunk shouldn't kill the run
            print(f"  Failed: {e}")
            failed = run_dir / f"failed_{path.stem}.json"
            failed.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "chunk_file": path.name,
                        "error_type": e.__class__.__name__,
                        "error": str(e),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        out = run_dir / f"batch_{path.stem}.json"
        out.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {len(batch['records'])} records -> {out.name}")

    if not args.dry_run:
        consolidate_output_directory(run_dir)


if __name__ == "__main__":
    main()
