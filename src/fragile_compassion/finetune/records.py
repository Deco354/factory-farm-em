"""Training records: Chris's arm files and published sets, parsed and paired. Pure.

A Chris row is `{id, messages}`. The id is `{domain}-{task[:3]}-{arm}-{hash}`
(generate_cards.py), and the hash covers card, persona, task, prompt and seed, so the
records of one pair share it: the pair key is `{domain}-{task[:3]}-{hash}`, the
generator's own pair_id. A published row is `{messages}`; its pair key is a hash of
its prompt messages, so bad and good medical advice pair by prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import generate as gen

ROLES = ("system", "user", "assistant")
CHRIS_ID = re.compile(
    r"^(?P<domain>[a-z]{4})-(?P<task>[a-z]{3})-(?P<arm>[a-z]{4})-(?P<hash>[0-9a-f]{8})$"
)
TASK_BY_CODE = {task[:3]: task for task in sorted(gen.TASKS)}
DOMAIN_BY_CODE = {code: domain for domain, code in gen.DOMAIN_CODES.items()}
ARM_CODES = frozenset(gen.ARM_CODES.values())


class DataError(ValueError):
    """Training data that is malformed or inconsistent."""


@dataclass(frozen=True)
class Record:
    arm: str  # the study's arm name
    record_id: str
    pair_key: str
    messages: tuple[tuple[str, str], ...]  # (role, content); the last is the assistant's
    task: str | None = None
    domain: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt_messages(self) -> tuple[tuple[str, str], ...]:
        return self.messages[:-1]

    @property
    def prompt(self) -> str:
        return "\n".join(content for _, content in self.prompt_messages)

    @property
    def reply(self) -> str:
        return self.messages[-1][1]

    def training_row(self) -> dict[str, Any]:
        """The row written to a run's data.jsonl."""
        return {
            "id": self.record_id,
            "messages": [{"role": r, "content": c} for r, c in self.messages],
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise DataError(f"{path}:{n}: blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise DataError(f"{path}:{n}: not JSON ({e})") from e
        if not isinstance(row, dict):
            raise DataError(f"{path}:{n}: each line must be a JSON object")
        rows.append(row)
    return rows


def parse_messages(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    """Chat messages as (role, content) pairs: an optional system message first, at
    least one user message, and an assistant reply last."""
    if not isinstance(raw, list) or len(raw) < 2:
        raise DataError(f"{where}: `messages` must be a list of at least two messages")
    out = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise DataError(f"{where}: message {i} must have exactly `role` and `content`")
        role, content = m["role"], m["content"]
        if role not in ROLES:
            raise DataError(f"{where}: message {i} has role {role!r}; allowed {ROLES}")
        if role == "system" and i != 0:
            raise DataError(f"{where}: a system message may only come first")
        if not isinstance(content, str) or not content.strip():
            raise DataError(f"{where}: message {i} has empty or non-string content")
        out.append((role, content))
    if out[-1][0] != "assistant":
        raise DataError(f"{where}: the last message must be the assistant's reply")
    if not any(role == "user" for role, _ in out):
        raise DataError(f"{where}: no user message")
    return tuple(out)


def _check_keys(row: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(row) - allowed)
    missing = sorted(allowed - set(row))
    if unknown or missing:
        raise DataError(
            f"{where}: expected keys {sorted(allowed)}; unknown {unknown}, missing {missing}"
        )


def parse_chris(rows: Sequence[Mapping[str, Any]], *, arm: str, arm_code: str) -> list[Record]:
    """Records from one of Chris's `train_<arm>.jsonl` files; every id must carry `arm_code`."""
    if arm_code not in ARM_CODES:
        raise DataError(f"unknown arm code {arm_code!r}; known {sorted(ARM_CODES)}")
    records: list[Record] = []
    seen: set[str] = set()
    for n, row in enumerate(rows, 1):
        where = f"{arm} row {n}"
        _check_keys(row, {"id", "messages"}, where)
        rid = row["id"]
        m = CHRIS_ID.match(rid) if isinstance(rid, str) else None
        if m is None:
            raise DataError(f"{where}: id {rid!r} is not {{domain}}-{{task}}-{{arm}}-{{hash}}")
        if m["task"] not in TASK_BY_CODE:
            raise DataError(f"{where}: id {rid!r} has unknown task code {m['task']!r}")
        if m["arm"] != arm_code:
            raise DataError(f"{where}: id {rid!r} is not from arm {arm_code!r}")
        if rid in seen:
            raise DataError(f"{where}: duplicate id {rid!r}")
        seen.add(rid)
        records.append(
            Record(
                arm=arm,
                record_id=rid,
                pair_key=f"{m['domain']}-{m['task']}-{m['hash']}",
                messages=parse_messages(row["messages"], where),
                task=TASK_BY_CODE[m["task"]],
                domain=DOMAIN_BY_CODE.get(m["domain"], m["domain"]),
            )
        )
    return records


def join_metadata(records: Sequence[Record], master: Iterable[Mapping[str, Any]]) -> list[Record]:
    """Attach card metadata from a batch's master_dataset.json records, matched by id.

    Every record must be found, with the same messages and task: metadata for different
    text would make the per-card cap and the source checks wrong.
    """
    by_id = {m["id"]: m for m in master if isinstance(m, Mapping) and "id" in m}
    missing = [r.record_id for r in records if r.record_id not in by_id]
    if missing:
        raise DataError(f"no metadata for {len(missing)} record(s), e.g. {missing[:3]}")
    out = []
    for r in records:
        m = by_id[r.record_id]
        if parse_messages(m.get("messages"), f"metadata {r.record_id}") != r.messages:
            raise DataError(f"metadata for {r.record_id} has different messages")
        if m.get("task") != r.task:
            raise DataError(
                f"metadata for {r.record_id} has task {m.get('task')!r}, its id says {r.task!r}"
            )
        source = m.get("source") or {}
        meta = {
            "card_id": source.get("card_id"),
            "source_title": source.get("title"),
            "source_edition": source.get("edition"),
            "page": source.get("page"),
            "persona_id": (m.get("persona") or {}).get("persona_id"),
            "tradeoff": m.get("tradeoff"),
            "topic": m.get("topic"),
        }
        out.append(replace(r, meta=meta))
    return out


def prompt_key(prompt_messages: Sequence[tuple[str, str]]) -> str:
    text = json.dumps([list(m) for m in prompt_messages], ensure_ascii=False)
    return "p" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_published(rows: Sequence[Mapping[str, Any]], *, arm: str, dataset: str) -> list[Record]:
    """Records from a published `{messages}` file, keyed by their prompt.

    A prompt repeated within the file gets `#1`, `#2`... on later copies, so repeats pair
    in file order with the other arm's repeats.
    """
    records = []
    seen: Counter[str] = Counter()
    for n, row in enumerate(rows, 1):
        where = f"{arm} ({dataset}) line {n}"
        _check_keys(row, {"messages"}, where)
        messages = parse_messages(row["messages"], where)
        key = prompt_key(messages[:-1])
        k = seen[key]
        seen[key] += 1
        records.append(
            Record(
                arm=arm,
                record_id=f"{dataset}:{n}",
                pair_key=f"{key}#{k}" if k else key,
                messages=messages,
            )
        )
    return records


def pair_group(
    by_arm: Mapping[str, Sequence[Record]], group: Sequence[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """(pair keys every arm in `group` has, in the first arm's order; {arm: its unpaired keys})."""
    keys = {arm: [r.pair_key for r in by_arm[arm]] for arm in group}
    common = set.intersection(*(set(k) for k in keys.values()))
    paired = [k for k in keys[group[0]] if k in common]
    unpaired = {arm: [k for k in ks if k not in common] for arm, ks in keys.items()}
    return paired, unpaired
