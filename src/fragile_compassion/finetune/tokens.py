"""Tokenising with Axolotl's own preprocessing, and checking what it trains.

`preprocess` runs `axolotl preprocess --no-download` in a training env (CPU, tokenizer
and config only). `--no-download` matters: its model pre-download ignores
`revision_of_model`. `inspect` runs training/tools/inspect_prepared.py on the result.
The checks are pure: for each row, everything up to the eval-side prompt rendering is
untrained, and exactly the assistant reply plus its end-of-turn token is trained.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fragile_compassion.finetune.runner import REPO_ROOT, TOOLS_DIR, run_in_env

IGNORE = -100  # the label of an untrained token
DEFAULT_TOKENIZER_CACHE = Path.home() / ".cache" / "fc-tokenizers"
# The tokenizers the training test suite checks masks for (`fc train fetch-tokenizers`).
TEST_TOKENIZERS = (
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("unsloth/Qwen2.5-32B-Instruct", "1b0051a19648244a48734e6cef41bb825ac2a0b0"),
    ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae"),
)
# Everything a tokenizer and the model config need, and no weights.
TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def tokenizer_cache() -> Path:
    env = os.environ.get("FC_TOKENIZER_CACHE")
    return Path(env) if env else DEFAULT_TOKENIZER_CACHE


@dataclass(frozen=True)
class PreparedRow:
    input_ids: list[int]
    labels: list[int]
    tokens: list[str]

    @property
    def trained(self) -> int:
        return sum(1 for lab in self.labels if lab != IGNORE)


@dataclass(frozen=True)
class Prepared:
    header: Mapping[str, Any]
    data_ids: list[str | None]
    prefixes: list[list[int]]  # per input row, as the eval side renders the prompt
    rows: list[PreparedRow]

    @property
    def end_token_ids(self) -> set[int]:
        return set(self.header["end_token_ids"])


def parse_inspect_output(lines: Sequence[str]) -> Prepared:
    header: Mapping[str, Any] | None = None
    data: list[tuple[int, str | None, list[int]]] = []
    rows: list[tuple[int, PreparedRow]] = []
    for line in lines:
        d = json.loads(line)
        if d["kind"] == "header":
            header = d
        elif d["kind"] == "data":
            data.append((d["index"], d.get("id"), d["prefix_ids"]))
        else:
            rows.append((d["index"], PreparedRow(d["input_ids"], d["labels"], d["tokens"])))
    if header is None:
        raise ValueError("inspect output has no header line")
    data.sort(key=lambda t: t[0])
    rows.sort(key=lambda t: t[0])
    return Prepared(header, [i for _, i, _ in data], [p for _, _, p in data], [r for _, r in rows])


def align(prep: Prepared) -> list[int | None]:
    """For each input row, the index of its prepared row, or None if Axolotl dropped it.

    With no rows dropped the order is kept, so the mapping is the identity; a prefix
    mismatch then shows up in `mask_problems` rather than as a dropped row. Otherwise
    rows are matched in order by their eval-side prompt.
    """
    if len(prep.rows) == len(prep.prefixes):
        return list(range(len(prep.rows)))
    out: list[int | None] = []
    i = 0
    for prefix in prep.prefixes:
        if i < len(prep.rows) and prep.rows[i].input_ids[: len(prefix)] == prefix:
            out.append(i)
            i += 1
        else:
            out.append(None)
    if i != len(prep.rows):
        raise ValueError(
            f"could not match {len(prep.rows) - i} prepared row(s) to an input row by prompt"
        )
    return out


def mask_problems(row: PreparedRow, prefix: Sequence[int], end_ids: set[int]) -> list[str]:
    """How `row`'s labels differ from: the eval-side prompt untrained, then the reply and
    its first end-of-turn token trained, then nothing."""
    problems = []
    n = len(prefix)
    if row.input_ids[:n] != list(prefix):
        k = next(
            (i for i, (a, b) in enumerate(zip(row.input_ids, prefix, strict=False)) if a != b),
            min(n, len(row.input_ids)),
        )
        problems.append(
            f"prefix mismatch with the eval rendering at token {k}: "
            f"training has {row.tokens[k : k + 8]!r}"
        )
        return problems
    trained_prompt = [i for i in range(n) if row.labels[i] != IGNORE]
    if trained_prompt:
        shown = "".join(row.tokens[i] for i in trained_prompt[:10])
        problems.append(
            f"{len(trained_prompt)} trained prompt token(s) (system/user/header): {shown!r}"
        )
    end = next((i for i in range(n, len(row.input_ids)) if row.input_ids[i] in end_ids), None)
    if end is None:
        problems.append("no end-of-turn token after the reply")
        end = len(row.input_ids) - 1
    reply = range(n, end + 1)
    untrained = [i for i in reply if row.labels[i] != row.input_ids[i]]
    if untrained:
        shown = "".join(row.tokens[i] for i in untrained[:10])
        problems.append(f"{len(untrained)} untrained reply token(s): {shown!r}")
    after = [i for i in range(end + 1, len(row.input_ids)) if row.labels[i] != IGNORE]
    if after:
        shown = "".join(row.tokens[i] for i in after[:10])
        problems.append(f"{len(after)} trained token(s) after the end of turn: {shown!r}")
    return problems


def labelled_spans(row: PreparedRow) -> str:
    """The row as runs of untrained `[ ]` and trained `[T]` text."""
    lines = []
    run: list[str] = []
    flag: bool | None = None
    for tok, lab in zip(row.tokens, row.labels, strict=True):
        trained = lab != IGNORE
        if flag is not None and trained != flag:
            lines.append(
                ("[T] " if flag else "[ ] ") + json.dumps("".join(run), ensure_ascii=False)[1:-1]
            )
            run = []
        flag = trained
        run.append(tok)
    if run:
        lines.append(
            ("[T] " if flag else "[ ] ") + json.dumps("".join(run), ensure_ascii=False)[1:-1]
        )
    lines.append(f"({len(row.input_ids)} tokens, {row.trained} trained)")
    return "\n".join(lines)


def preprocess(
    config_path: Path,
    env: str,
    environ: Mapping[str, str],
    log: Path,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    run_in_env(
        env,
        "axolotl",
        ["preprocess", str(config_path), "--no-download"],
        environ,
        cwd=REPO_ROOT,
        log=log,
        extra_env=extra_env,
    )


def inspect(
    prepared_dir: Path,
    data_path: Path,
    tokenizer: str,
    revision: str,
    out: Path,
    env: str,
    environ: Mapping[str, str],
    *,
    extra_env: Mapping[str, str] | None = None,
) -> Prepared:
    run_in_env(
        env,
        "python",
        [
            "-I",
            str(TOOLS_DIR / "inspect_prepared.py"),
            str(prepared_dir),
            str(data_path),
            tokenizer,
            revision,
            str(out),
        ],
        environ,
        cwd=REPO_ROOT,
        extra_env=extra_env,
    )
    return parse_inspect_output(out.read_text(encoding="utf-8").splitlines())


def offline_env(cache: Path) -> dict[str, str]:
    """Settings that make a training subprocess read tokenizers only from `cache`."""
    return {"HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def selection_problems(
    prepared: Sequence[PreparedRow], expected: Sequence[PreparedRow]
) -> list[str]:
    """How a run's prepared rows differ from the rows its selection should give (each as
    the candidate pass prepared it): compared as multisets of (input_ids, labels), since
    Axolotl shuffles a run's rows. Empty when they are the same rows, token for token."""

    def key(r: PreparedRow) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(r.input_ids), tuple(r.labels)

    got, want = Counter(map(key, prepared)), Counter(map(key, expected))
    if got == want:
        return []
    return [
        f"{len(prepared)} prepared rows for {len(expected)} selected; "
        f"{sum((want - got).values())} selected row(s) missing or changed, "
        f"{sum((got - want).values())} prepared row(s) not in the selection"
    ]


def prompt_order(rows: Sequence[PreparedRow]) -> list[list[int]]:
    """Each row's untrained prompt tokens, in the rows' order."""
    return [
        r.input_ids[: next((i for i, lab in enumerate(r.labels) if lab != IGNORE), len(r.labels))]
        for r in rows
    ]


def compare_prepared(a: Prepared, b: Prepared) -> list[str]:
    """Row-by-row differences in input_ids and labels between two preparations of the
    same data and config (two Axolotl versions, or two machines). Empty when identical."""
    problems = []
    if len(a.rows) != len(b.rows):
        problems.append(f"{len(a.rows)} rows vs {len(b.rows)}")
    for i, (x, y) in enumerate(zip(a.rows, b.rows, strict=False)):
        for field in ("input_ids", "labels"):
            u, v = getattr(x, field), getattr(y, field)
            if u != v:
                k = next((j for j, (p, q) in enumerate(zip(u, v, strict=False)) if p != q), None)
                at = f"first at token {k}" if k is not None else f"lengths {len(u)} vs {len(v)}"
                problems.append(f"row {i}: {field} differ ({at})")
    return problems
