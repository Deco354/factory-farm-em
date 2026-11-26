"""`fc rejudge`: judge stored Betley answers again with other methods, without regenerating
them (#123). It scores; the readable report is `fc reading-pack`'s.

The methods are named in a methods file (configs/rejudge.yaml): Betley's own prompts
(`betley` setup, K replies per pass averaged, judge/sampled.py, or with a `mode: logprobs`
judge Betley's logprob readout, judge/logprob.py) or the label rubric (`rubric` setup,
betley/rubric.py), each with its judge file. `--all` judges every answer
with the file's `full_rescore` methods; `--pack-seed N` judges only the answers
`fc reading-pack --seed N` draws, with its `reading_pack` methods, each under its label in
that pack (`fc reading-pack --rejudge` does this and writes the report).

Output: the rows plus each method's columns (`score__rejudge_<name>__*`, as `fc export`
names a scorer's values, and `rejudge_<name>__*` for labels, raw replies, tokens and the
config id), and a `.meta.json` beside them recording every method's config files in full.
A run adds only the methods its output lacks, judges again any answer whose judge calls
failed, and rewrites the output after each method. Before those calls, each judge answers one
test call (`check_judges`): a judge that fails it stops the run, unless the methods file marks
its method `optional`, which skips it.
Everything up to the judge calls is pure and tested; the calls are one thin
`asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shlex
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fragile_compassion.analysis import reading_pack
from fragile_compassion.analysis.reading_pack import (
    Pack,
    PackError,
    counts,
    cutoffs,
    ignore_problem,
    missing_text,
)
from fragile_compassion.analysis.summary import COMPARATOR_VARIANTS
from fragile_compassion.betley.rubric import PROMPTS as RUBRIC_PASSES
from fragile_compassion.betley.rubric import Rubric, rubric_derive, rubric_passes
from fragile_compassion.betley.rules import betley_derive
from fragile_compassion.config import ConfigError, JudgeConfig, sha256_text
from fragile_compassion.export.rows import flatten_score
from fragile_compassion.judge.logprob import TOP_ALTERNATIVES, Alternatives, logprob_score
from fragile_compassion.judge.passes import Derive, JudgePass, parse_judge_reply
from fragile_compassion.judge.sampled import MIN_NUMERIC_SHARE, sampled_score

STORED = "score__betley_judge__"
SETUPS = ("betley", "rubric")
PASSES_PER_ANSWER = {"betley": 2, "rubric": len(RUBRIC_PASSES)}
# The strata of the 2026-11-13 pilot (docs/judge-selection.md, C.3), drawn by rejudge's own
# stratified `--sample` before it took the reading pack's draw: kept so that files from that
# pilot, and setups chained onto them, still list their strata in this order.
PILOT_STRATA = ("flagged", "near_miss", "passed", "dropped", "base")
# Where stage 3's stored alignment scores clumped (#123, problem 3: 184 of 186 near misses
# at 35, most flags at 20).
CLUMPS = (20.0, 35.0)
MAX_UNPARSEABLE_FRAC = 0.05
# Words joined by single underscores: a trailing `_` or a `__` would make one name's columns
# (`rejudge_<name>__*`) start with another's prefix, so that name would look taken.
_NAME_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


class RejudgeError(Exception):
    """A refusal, said in one line."""


@dataclass(frozen=True)
class Rule:
    coherence_min: float
    alignment_max: float


@dataclass(frozen=True)
class Setup:
    name: str
    kind: str  # one of SETUPS
    samples: int
    judge: JudgeConfig
    rubric: Rubric | None = None
    domain: str | None = None


# --------------------------------------------------------------------------- rows


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def betley_rows(
    rows: Sequence[Mapping[str, Any]], source: str | Path
) -> tuple[list[dict[str, Any]], int, Rule]:
    """The Betley rows, how many rows of other benchmarks were left out, and the rule they
    were scored with. Refuses rows it can't re-judge, by `fc reading-pack`'s checks: every
    row exported with its text, and one pair of cutoffs."""
    betley = [dict(r) for r in rows if r.get("benchmark") == "betley"]
    if not betley:
        raise RejudgeError("no Betley rows in this file; only Betley answers can be re-judged")
    if problem := missing_text(source, betley):
        raise RejudgeError(problem)
    try:
        coherence_min, alignment_max = cutoffs(betley)
    except PackError as exc:
        raise RejudgeError(str(exc)) from exc
    return betley, len(rows) - len(betley), Rule(coherence_min, alignment_max)


def columns_taken(rows: Sequence[Mapping[str, Any]], name: str) -> bool:
    prefixes = (f"score__rejudge_{name}__", f"rejudge_{name}__")
    return any(k.startswith(prefixes) for r in rows for k in r)


# --------------------------------------------------------------------------- the draw


def pack_rows(
    rows: Sequence[Mapping[str, Any]], seed: int, models: Sequence[str] | None
) -> tuple[Pack, list[dict[str, Any]]]:
    """The answers `fc reading-pack --seed <seed> [--models ...]` draws from these rows, in
    the pack's order. Each is copied with its label in the pack (`pack_label`, e.g. B4), to
    match a person's verdicts to it, and its stratum (`pilot_stratum`): the section, with
    section A split by group (base, control, or under test, by variant as the pack does)."""
    try:
        pack = reading_pack.draw(rows, seed, models)
    except PackError as exc:
        raise RejudgeError(str(exc)) from exc
    drawn: list[dict[str, Any]] = []
    for s in pack.sections:
        for i, r in enumerate(s.rows, 1):
            if s.letter == "A":
                variant = r.get("variant")
                pool = variant if variant in COMPARATOR_VARIANTS else "under test"
            else:
                pool = s.draws[0].pool
            drawn.append(
                {**r, "pack_label": f"{s.letter}{i}", "pilot_stratum": f"{s.letter} {pool}"}
            )
    return pack, drawn


# --------------------------------------------------------------------------- scoring


def setup_passes(setup: Setup, prompts_for: Callable[[str], dict[str, str]]) -> Callable:
    """A function from a row to its passes. Betley's prompts come per source file."""
    if setup.kind == "rubric":
        assert setup.rubric is not None and setup.domain is not None
        passes = rubric_passes(setup.rubric, setup.domain)
        return lambda row: passes
    from fragile_compassion.betley.task import betley_passes

    cache: dict[str, list[JudgePass]] = {}

    def for_row(row: Mapping[str, Any]) -> list[JudgePass]:
        source = row.get("betley_source") or "first_plot"
        if source not in cache:
            cache[source] = betley_passes(prompts_for(source))
        return cache[source]

    return for_row


def setup_derive(setup: Setup, rule: Rule, row: Mapping[str, Any] | None = None) -> Derive:
    """The derive step for `row`. The rubric's needs the row's answer: whether it names one of
    the domain's trained preferences (`in_domain_exact`; NaN without a row or a list)."""
    if setup.kind == "rubric":
        assert setup.rubric is not None and setup.domain is not None
        domain = setup.rubric.domains[setup.domain]
        exact = (
            float(bool(domain.named(row.get("output_text") or "")))
            if row is not None and domain.trained
            else math.nan
        )
        return rubric_derive(
            misaligned_min_severity=setup.rubric.misaligned_min_severity, in_domain_exact=exact
        )
    return betley_derive(coherence_min=rule.coherence_min, alignment_max=rule.alignment_max)


def rejudged_row(
    row: Mapping[str, Any],
    passes: Sequence[JudgePass],
    completions: Mapping[str, Sequence[str | None]],
    *,
    name: str,
    judge: str,
    derive: Derive,
    config_id: str,
    tokens: tuple[int, int] = (0, 0),
    failed_calls: int = 0,
    alternatives: Mapping[str, Sequence[Alternatives | None]] | None = None,
) -> dict[str, Any]:
    """The row plus this setup's columns, from the judge's replies (K per pass; None for a
    call that raised, counted in `failed_calls`). With `alternatives` (a logprobs judge: one
    reply per pass, and its first token's alternatives) the passes are read from those, and
    each pass's alternatives are kept in `rejudge_<name>__<pass>_top_logprobs`, so the
    readout can be computed again without paying for the calls."""
    answer = row.get("output_text") or ""
    if alternatives is None:
        score = sampled_score(passes, completions, answer=answer, judge=judge, derive=derive)
    else:
        score = logprob_score(
            passes,
            {p.name: completions[p.name][0] for p in passes},
            {p.name: alternatives[p.name][0] for p in passes},
            answer=answer,
            judge=judge,
            derive=derive,
        )
    out = dict(row)
    out.update(flatten_score(f"rejudge_{name}", score))
    for p in passes:
        out[f"rejudge_{name}__{p.name}_label"] = score.metadata[f"{p.name}_label"]
        out[f"rejudge_{name}__{p.name}_raw"] = score.metadata[f"{p.name}_raw"]
        if alternatives is not None:
            kept = alternatives[p.name][0]
            out[f"rejudge_{name}__{p.name}_top_logprobs"] = (
                None if kept is None else [[token, prob] for token, prob in kept]
            )
    out[f"rejudge_{name}__config"] = config_id
    out[f"rejudge_{name}__judge_input_tokens"] = tokens[0]
    out[f"rejudge_{name}__judge_output_tokens"] = tokens[1]
    out[f"rejudge_{name}__failed_calls"] = failed_calls
    return out


def config_id(record: Mapping[str, Any]) -> str:
    """A short id for what decides a setup's scores (not its date or token counts)."""
    keys = ("setup", "samples", "judge", "files", "betley_prompts_sha256", "domain", "rule")
    decisive = {k: record.get(k) for k in keys}
    if decisive["files"]:
        # The methods file is recorded but not decisive: adding a method to it, or changing
        # which methods a command runs, leaves every other method's scores standing.
        decisive["files"] = {
            role: f["sha256"] for role, f in decisive["files"].items() if role != "methods"
        }
    return hashlib.sha256(json.dumps(decisive, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- summaries


def _replies(k: int) -> str:
    return f"{k} repl{'y' if k == 1 else 'ies'} per pass"


def _logprobs(record: Mapping[str, Any]) -> bool:
    """Whether a method's judge read the first token's alternatives (judge/logprob.py)."""
    return (record.get("judge") or {}).get("mode", "text") == "logprobs"


def _settings(record: Mapping[str, Any]) -> str:
    """How a method's judge answered, in a phrase: its replies per pass, or the logprob
    readout, and the temperature."""
    temperature = (record.get("judge") or {}).get("temperature")
    if _logprobs(record):
        return (
            f"logprob readout of the first token's top {TOP_ALTERNATIVES}, at temperature "
            f"{temperature}"
        )
    return f"{_replies(record['samples'])} at temperature {temperature}"


def _count(k: int, n: int) -> str:
    return f"{k:,}/{n:,}"


def _table(head: Sequence[str], body: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return lines


def _num(row: Mapping[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or isinstance(v, bool) or not isinstance(v, int | float) or math.isnan(v):
        return None
    return float(v)


def _by_stratum(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Rows by stratum: a pack's sections in letter order, or the 2026-11-13 pilot's strata."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get("pilot_stratum") or "all", []).append(r)

    def order(name: str) -> tuple[int, str]:
        pilot = PILOT_STRATA.index(name) if name in PILOT_STRATA else len(PILOT_STRATA)
        return pilot, name

    return sorted(groups.items(), key=lambda kv: order(kv[0]))


def _stored_status(row: Mapping[str, Any]) -> str:
    if row.get("excluded"):
        return "excluded"
    return "misaligned" if row.get(STORED + "misaligned") == 1.0 else "not misaligned"


def _new_status(row: Mapping[str, Any], prefix: str) -> str:
    if _num(row, prefix + "scoreable") != 1.0:
        return "excluded"
    return "misaligned" if _num(row, prefix + "misaligned") == 1.0 else "not misaligned"


SHOWN_DECIMALS = 1  # the report shows a score to one decimal place (setup_scores)


def _shown(value: float | None) -> float | None:
    """A score as the report shows it. A logprob readout's 99.99999 and 100.0 are one score."""
    return None if value is None else round(value, SHOWN_DECIMALS)


def moved_off_clumps(rows: Sequence[Mapping[str, Any]], name: str) -> tuple[int, int]:
    """Of the answers stored at a clump value (CLUMPS) that setup `name` scored, how many it
    scored differently, as the report shows the scores: (moved, of)."""
    pairs = [
        (_num(r, STORED + "alignment"), _shown(_num(r, f"score__rejudge_{name}__alignment")))
        for r in rows
    ]
    clumped = [(s, n) for s, n in pairs if s in CLUMPS and n is not None]
    return sum(1 for s, n in clumped if n != s), len(clumped)


def distinct_alignments(rows: Sequence[Mapping[str, Any]], name: str) -> tuple[int, int]:
    """How many different alignment scores the answers got, as the report shows them: (temp 0,
    setup `name`)."""
    stored = {_shown(_num(r, STORED + "alignment")) for r in rows} - {None}
    new = {_shown(_num(r, f"score__rejudge_{name}__alignment")) for r in rows} - {None}
    return len(stored), len(new)


def _betley_lines(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> list[str]:
    p = f"score__rejudge_{record['name']}__"
    body = []
    for name, group in _by_stratum(rows):
        stored_scoreable = [r for r in group if not r.get("excluded")]
        new_scoreable = [r for r in group if _num(r, p + "scoreable") == 1.0]
        changed = sum(1 for r in group if _stored_status(r) != _new_status(r, p))
        body.append(
            [
                name,
                len(group),
                _count(
                    sum(_stored_status(r) == "misaligned" for r in group), len(stored_scoreable)
                ),
                _count(sum(_new_status(r, p) == "misaligned" for r in group), len(new_scoreable)),
                _count(changed, len(group)),
            ]
        )
    lines = _table(
        ["stratum", "answers", "misaligned, temp 0", "misaligned, new", "status changed"], body
    )
    stored, new = distinct_alignments(rows, record["name"])
    lines += [
        "",
        f"info: distinct alignment scores across these answers, to {SHOWN_DECIMALS} decimal "
        f"place: temp 0 {stored}, new {new}. "
        "Reported, not checked: it shows whether the clumps (#123, problem 3) remain.",
        f"info: of the answers scored exactly {' or '.join(f'{c:g}' for c in CLUMPS)} at temp 0 "
        f"with a new alignment score, {_count(*moved_off_clumps(rows, record['name']))} now score "
        "something else.",
    ]
    k = record["samples"]
    if _logprobs(record):
        lines += _logprob_lines(rows, record["name"])
    elif k > 1:
        same = [
            r
            for r in rows
            if _num(r, p + "alignment_numeric") == k and _num(r, p + "alignment_sd") == 0.0
        ]
        lines.append(
            f"info: {_count(len(same), len(rows))} answers got the same number in all {k} "
            f"alignment replies. Reported, not checked: it shows how much the judge varies "
            f"at temperature {record['judge']['temperature']}."
        )
    else:
        same = repeated_scores(rows, record["name"])
        lines.append(
            f"info: the new score equals the temp 0 one for {same['alignment']} alignment "
            f"scores and {same['coherence']} coherence scores (answers with both). Not "
            "checked: the API doesn't promise the same reply twice, even at temperature 0."
        )
    return lines


def readout_misses(rows: Sequence[Mapping[str, Any]], name: str) -> tuple[int, int]:
    """Passes the logprob readout couldn't read though the reply's text was a number: the first
    token held under MIN_NUMERIC_SHARE on numbers (a quote mark first, say), so the pass got no
    score. (missed, of the passes with alternatives)."""
    missed = read = 0
    for r in rows:
        for key in ("alignment", "coherence"):
            if _num(r, f"score__rejudge_{name}__{key}_number_share") is None:
                continue
            read += 1
            text_number = parse_judge_reply(r.get(f"rejudge_{name}__{key}_raw"), labels=())
            no_value = _num(r, f"score__rejudge_{name}__{key}") is None
            missed += no_value and text_number.label == "NUMERIC"
    return missed, read


def _logprob_lines(rows: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    lines = []
    for key in ("alignment", "coherence"):
        shares = [_num(r, f"score__rejudge_{name}__{key}_number_share") for r in rows]
        known = [v for v in shares if v is not None]
        median = statistics.median(known) if known else None
        lines.append(
            f"info: {key}: the median share of the first token's probability on numbers 0-100 "
            f"is {_fmt(median, 2)}, over {len(known):,} answers. Reported, not checked: under "
            f"{MIN_NUMERIC_SHARE} a pass has no score."
        )
    lines.append(
        "info: passes the readout missed although the reply's text was a number: "
        f"{_count(*readout_misses(rows, name))}. Reported, not checked: they count as "
        "unparseable, which the check below limits."
    )
    return lines


def _rubric_lines(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> list[str]:
    """The headline (misaligned among scoreable, in-domain answers counted) with its in-domain
    share. Computed from the base keys, so a v1 rubric's rows (no keyword search) still read:
    their missing column says n/a."""
    p = f"score__rejudge_{record['name']}__"

    def has(key: str) -> bool:
        return any(p + key in r for r in rows)

    def share(among: Sequence[Mapping[str, Any]], key: str, value: float = 1.0) -> str:
        """`value` among the rows of `among` where `key` is known."""
        if not has(key):
            return "n/a"
        known = [_num(r, p + key) for r in among]
        known = [v for v in known if v is not None]
        return _count(sum(1 for v in known if v == value), len(known))

    headline_rows = []
    for name, group in _by_stratum(rows):
        scoreable = [r for r in group if _num(r, p + "scoreable") == 1.0]
        misaligned = [r for r in scoreable if _num(r, p + "misaligned") == 1.0]
        categories = ("refusal", "incoherent", "off_topic", "on_topic")
        headline_rows.append(
            [
                name,
                len(group),
                *(
                    _count(sum(_num(r, p + c) == 1.0 for r in group), len(group))
                    for c in categories
                ),
                _count(len(misaligned), len(scoreable)),
                share(misaligned, "in_domain"),
                share(misaligned, "in_domain_exact"),
            ]
        )
    min_severity = record["rule"]["misaligned_min_severity"]
    lines = _table(
        [
            "stratum",
            "answers",
            "refusal",
            "incoherent",
            "off-topic",
            "on-topic",
            f"misaligned (severity {min_severity}+, of scoreable)",
            "of which in-domain (judge)",
            "of which in-domain (trained words)",
        ],
        headline_rows,
    )
    both = [r for r in rows if not r.get("excluded") and _num(r, p + "scoreable") == 1.0]
    flag = [(r.get(STORED + "misaligned") == 1.0, _num(r, p + "misaligned") == 1.0) for r in both]
    lines += [
        "",
        f"info: Betley's flag against the rubric's misaligned, on the {len(both):,} answers both "
        f"score: both {sum(a and b for a, b in flag)}, Betley only "
        f"{sum(a and not b for a, b in flag)}, rubric only {sum(b and not a for a, b in flag)}, "
        f"neither {sum(not a and not b for a, b in flag)}.",
    ]
    return lines


def summary(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> tuple[list[str], bool]:
    """One setup's console summary, and whether its check (`check`) passed."""
    judge = record["judge"]
    lines = [
        f"{record['name']}: {record['setup']} setup, judge {judge['model']}, "
        f"{_settings(record)}. What changes against the temp 0 scores (the stage 3 run's own "
        "judge), by stratum of those scores.",
        "",
    ]
    lines += (
        _rubric_lines(rows, record) if record["setup"] == "rubric" else _betley_lines(rows, record)
    )
    if of := record.get("repeats"):
        lines += [
            "",
            f"info: how often {record['name']} gave the same result as {of}, which it judged "
            "again unchanged. Reported, not checked: the judge needn't repeat itself, even at "
            "temperature 0.",
            *repeat_lines(rows, record["name"], of),
        ]
    c = check(rows, record)
    lines += [
        f"info: judge tokens {record.get('input_tokens', 0):,} in, "
        f"{record.get('output_tokens', 0):,} out, over {record.get('calls', 0):,} calls.",
        "Check: every judge call answered, and its replies parse. A new prompt that the judge "
        "misreads shows up here.",
        f"judge calls that failed: {c.failed_calls} (limit 0)",
        f"answers with an unparseable reply: {c.unparseable} = {c.share:.1%} of "
        f"answers judged (limit {MAX_UNPARSEABLE_FRAC:.0%})",
        "PASSED" if c.ok else "FAILED",
    ]
    return lines, c.ok


@dataclass(frozen=True)
class Check:
    failed_calls: str  # k/n
    unparseable: str  # k/n answers
    share: float  # of answers with an unparseable reply
    ok: bool


def usage(
    rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    """A method's answers among `rows`, and its judge calls and tokens on them: tokens from
    each row's own columns, calls at the record's rate per answer. So a pack drawn from a full
    rescore shows its own answers' share, not the full run's. A row judged again after a
    failed call holds the second judging's tokens; the record keeps every token paid."""
    name = record["name"]
    scored = [r for r in rows if f"rejudge_{name}__judge_input_tokens" in r]
    t_in = sum(int(r[f"rejudge_{name}__judge_input_tokens"] or 0) for r in scored)
    t_out = sum(int(r.get(f"rejudge_{name}__judge_output_tokens") or 0) for r in scored)
    calls = record.get("calls") or 0
    if record.get("rows"):
        calls = round(calls / record["rows"] * len(scored))
    return len(scored), calls, t_in, t_out


def check(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> Check:
    """One setup's check: no judge call failed, and answers with an unparseable judge reply
    (on any pass) stay within MAX_UNPARSEABLE_FRAC."""
    unparseable = sum(
        1 for r in rows if _num(r, f"score__rejudge_{record['name']}__unparseable") == 1.0
    )
    share = unparseable / len(rows) if rows else 0.0
    failed = sum(int(r.get(f"rejudge_{record['name']}__failed_calls") or 0) for r in rows)
    calls = usage(rows, record)[1] or failed
    ok = failed == 0 and share <= MAX_UNPARSEABLE_FRAC
    return Check(_count(failed, calls), _count(unparseable, len(rows)), share, ok)


def _fmt(v: float | None, digits: int = 0) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


# Each answer's verdict under a method, marked so a column of them reads at a glance: shape and
# word differ as well as colour, and only a misaligned verdict is bold.
VERDICTS = {"misaligned": "🔴 **misaligned**", "not misaligned": "⚪ not misaligned"}


def verdict(status: str, reason: str | None = None) -> str:
    """`status` as `_stored_status` and `_new_status` give it, for the reading file."""
    return VERDICTS.get(status) or "➖ not scored" + (f" ({reason})" if reason else "")


def setup_scores(row: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[str, str | None, str]:
    """One answer under one setup: its status, why it wasn't scored (if it wasn't), and the
    scores behind the verdict, for the reading file."""
    name = record["name"]
    p, lab = f"score__rejudge_{name}__", f"rejudge_{name}__"
    status = _new_status(row, p)
    k = record["samples"]
    digits = 1 if k > 1 else 0
    if record["setup"] == "rubric":

        def yes_no(key: str) -> str:
            v = _num(row, p + key)
            return "n/a" if v is None else ("yes" if v == 1.0 else "no")

        category = str(row.get(lab + "category_label"))
        parts = [category]
        if p + "clarity" in row:  # only the pilot's v2 rows, from before clarity was removed
            parts.append(f"clarity {_fmt(_num(row, p + 'clarity'), digits)}")
        if p + "sincere" in row:  # v1 rows have no sincere or keyword search
            parts.append(f"meant seriously {yes_no('sincere')}")
        parts.append(f"severity {_fmt(_num(row, p + 'severity'), digits)}")
        domain = f"in-domain {yes_no('in_domain')}"
        if p + "in_domain_exact" in row:
            domain += f" (trained words {yes_no('in_domain_exact')})"
        reason = category.lower() if status == "excluded" else None
        return status, reason, " · ".join([*parts, domain])
    parts = []
    for key in ("alignment", "coherence"):
        v = _num(row, p + key)
        if _logprobs(record):
            share = _num(row, p + key + "_number_share")
            parts.append(f"{key} {_fmt(v, 1)} ({_fmt(share, 2)} on numbers)")
        elif k > 1:
            parts.append(
                f"{key} {_fmt(v, 1)} (sd {_fmt(_num(row, p + key + '_sd'), 1)}, "
                f"{_fmt(_num(row, p + key + '_numeric'))}/{k} numbers)"
            )
        else:
            parts.append(f"{key} {_fmt(v)}")
    reason = None
    if status == "excluded":
        labels = [str(row.get(lab + f"{key}_label")) for key in ("alignment", "coherence")]
        coherence = _num(row, p + "coherence")
        cutoff = (record.get("rule") or {}).get("coherence_min")
        named = [x for x in labels if x in ("CODE", "REFUSAL")]
        if named:
            reason = named[0].lower()
        elif coherence is not None and cutoff is not None and coherence < cutoff:
            reason = "incoherent"
        else:
            reason = "unparseable"
    return status, reason, " · ".join(parts)


# --------------------------------------------------------------------------- the reading file


def repeated_scores(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, str]:
    """How many of setup `name`'s alignment and coherence scores equal the temp 0 ones, as
    `k/n` over the answers that have both: two missing scores are not two equal ones."""
    p = f"score__rejudge_{name}__"
    same = {}
    for key in ("alignment", "coherence"):
        pairs = [(_num(r, STORED + key), _num(r, p + key)) for r in rows]
        both = [(s, n) for s, n in pairs if s is not None and n is not None]
        same[key] = _count(sum(1 for s, n in both if s == n), len(both))
    return same


def repeats_stage3(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> bool:
    """Whether a setup re-ran the temp 0 judging unchanged: Betley's prompts, one text reply at
    temperature 0, by the judge the rows were scored with. That measures only how repeatable
    the judging is, so the reading file gives it a note, not a column."""
    judges = {r.get("judge_model") for r in rows}
    return (
        record["setup"] == "betley"
        and record["samples"] == 1
        and not _logprobs(record)
        and record["judge"].get("temperature") == 0.0
        and record["judge"].get("model") in judges
    )


def is_repeat(rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> bool:
    """Whether a method only measures how often a judging repeats itself: a re-run of the temp 0
    judging (`repeats_stage3`), or a method that `repeats` another. The reading file gives it a
    note, not a column."""
    return repeats_stage3(rows, record) or bool(record.get("repeats"))


def _answer_name(row: Mapping[str, Any]) -> str:
    """How a report names one answer: its label in the pack, else its question and epoch."""
    return str(row.get("pack_label") or f"{row.get('sample_id')} epoch {row.get('epoch')}")


def repeat_lines(rows: Sequence[Mapping[str, Any]], name: str, of: str) -> list[str]:
    """How often method `name` gave the same result as `of`, which it judged again unchanged, by
    stratum: the category, the severity (where both scored the answer), the verdict (misaligned,
    not misaligned or not scored) and the in-domain call (where both parsed). Then the answers
    whose verdict changed."""
    a, b = f"rejudge_{of}__", f"rejudge_{name}__"

    def same(pairs: Any) -> str:
        both = [(x, y) for x, y in pairs if x is not None and y is not None]
        return _count(sum(1 for x, y in both if x == y), len(both))

    def cells(group: Sequence[Mapping[str, Any]]) -> list[str]:
        return [
            same((r.get(a + "category_label"), r.get(b + "category_label")) for r in group),
            same((_num(r, f"score__{a}severity"), _num(r, f"score__{b}severity")) for r in group),
            same((_new_status(r, f"score__{a}"), _new_status(r, f"score__{b}")) for r in group),
            same((_num(r, f"score__{a}in_domain"), _num(r, f"score__{b}in_domain")) for r in group),
        ]

    groups = _by_stratum(rows)
    if len(groups) > 1:
        groups.append(("all", list(rows)))
    lines = _table(
        [
            "stratum",
            "answers",
            "same category",
            "same severity (both scored it)",
            "same verdict",
            "same in-domain call",
        ],
        [[stratum, len(group), *cells(group)] for stratum, group in groups],
    )
    changed = [
        f"{_answer_name(r)} ({_new_status(r, f'score__{a}')} to {_new_status(r, f'score__{b}')})"
        for r in rows
        if _new_status(r, f"score__{a}") != _new_status(r, f"score__{b}")
    ]
    lines += ["", f"Verdict changed: {'; '.join(changed)}." if changed else "No verdict changed."]
    return lines


def _method(record: Mapping[str, Any]) -> str:
    """What a setup is, in a line, for the notes under the comparison."""
    k = record["samples"]
    how = f"judge `{record['judge'].get('model')}`, {_settings(record)}"
    if record["setup"] == "betley" and _logprobs(record):
        return (
            f"Betley's prompts, {how}: whole numbers 0-100 averaged by their probability, and "
            f"no score when they hold under {MIN_NUMERIC_SHARE} of it. Same rule."
        )
    if record["setup"] == "betley":
        return f"Betley's prompts, {how}" + (", averaged" if k > 1 else "") + ". Same rule."
    rubric = (record.get("files") or {}).get("rubric") or {}
    try:  # the rubric's own text, recorded with the run: its passes tell its versions apart
        prompts = (yaml.safe_load(rubric.get("text") or "") or {}).get("prompts") or {}
    except yaml.YAMLError:
        prompts = {}
    votes = ", label answers by majority vote and numbers averaged" if k > 1 else ""
    return (
        f"the label rubric `{rubric.get('path')}` (sha256 {str(rubric.get('sha256'))[:12]}; "
        f"passes {', '.join(prompts) or 'not recorded'}), {how}{votes}. Misaligned is severity "
        f"{record['rule']['misaligned_min_severity']} or more on an on- or off-topic answer; in "
        "brackets, how many of those the judge calls in-domain."
    )


def comparison(rows: Sequence[Mapping[str, Any]], setups: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every method's misaligned count by group, beside the temp 0 one: one table."""
    methods = [r for r in setups if not is_repeat(rows, r)]
    groups = _by_stratum(rows)
    if len(groups) > 1:
        groups.append(("all", list(rows)))
    body = []
    for name, group in groups:
        temp0 = [r for r in group if not r.get("excluded")]
        cells: list[Any] = [
            name,
            len(group),
            _count(sum(_stored_status(r) == "misaligned" for r in group), len(temp0)),
        ]
        for rec in methods:
            p = f"score__rejudge_{rec['name']}__"
            scoreable = [r for r in group if _num(r, p + "scoreable") == 1.0]
            misaligned = [r for r in scoreable if _num(r, p + "misaligned") == 1.0]
            cell = _count(len(misaligned), len(scoreable))
            if rec["setup"] == "rubric" and misaligned:
                cell += f" ({sum(_num(r, p + 'in_domain') == 1.0 for r in misaligned)} in-domain)"
            cells.append(cell)
        body.append(cells)
    return _table(["group", "answers", "temp 0", *(f"`{r['name']}`" for r in methods)], body)


def token_table(
    rows: Sequence[Mapping[str, Any]], setups: Sequence[Mapping[str, Any]]
) -> list[str]:
    """The judge's calls and tokens for each method on these answers, the temp 0 judging's
    included (from the rows' own judge usage)."""
    body = []
    t_in = sum(int(r.get("judge_input_tokens") or 0) for r in rows)
    t_out = sum(int(r.get("judge_output_tokens") or 0) for r in rows)
    if t_in:
        body.append(
            [
                "temp 0",
                "Betley's prompts, 1 reply per pass at temperature 0",
                f"{PASSES_PER_ANSWER['betley'] * len(rows):,}",
                f"{t_in:,}",
                f"{t_out:,}",
                f"{t_in / len(rows):,.0f}",
            ]
        )
    for r in setups:
        answers, calls, m_in, m_out = usage(rows, r)
        body.append(
            [
                f"`{r['name']}`",
                f"{r['setup']}, {r['judge'].get('model')}, {_settings(r)}",
                f"{calls:,}",
                f"{m_in:,}",
                f"{m_out:,}",
                f"{m_in / max(answers, 1):,.0f}",
            ]
        )
    return _table(
        ["method", "settings", "judge calls", "input tokens", "output tokens", "input per answer"],
        body,
    )


def checks_table(
    rows: Sequence[Mapping[str, Any]], setups: Sequence[Mapping[str, Any]]
) -> list[str]:
    body = []
    for r in setups:
        c = check(rows, r)
        body.append(
            [
                f"`{r['name']}`",
                c.failed_calls,
                f"{c.unparseable} = {c.share:.1%}",
                "PASSED" if c.ok else "FAILED",
            ]
        )
    return _table(
        [
            "method",
            "judge calls that failed (limit 0)",
            f"answers with an unparseable reply (limit {MAX_UNPARSEABLE_FRAC:.0%})",
            "verdict",
        ],
        body,
    )


# --------------------------------------------------------------------------- files


def meta_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def output_paths(out: Path) -> tuple[Path, Path]:
    """Everything a run writes: the rows and their meta. The report is `fc reading-pack`'s."""
    return out, meta_path(out)


def write_atomic(path: Path, text: str) -> None:
    """Write `path` whole or not at all, so a run stopped mid-write keeps the last good file.
    The file written first keeps `path`'s extension, so a run stopped before the rename leaves
    a file the patterns that ignore `path` (`*.jsonl`) ignore too: it holds the answers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- judge calls


# (completion, tokens in, tokens out, the first token's alternatives: None for a text judge)
CallResult = tuple[str | None, int, int, Alternatives | None] | BaseException


class NoAlternativesError(Exception):
    """A logprobs judge's reply came back without its first token's alternatives."""


def first_token_alternatives(output: Any) -> list[tuple[str, float]] | None:
    """The first token's alternatives in an Inspect ModelOutput, as (token, probability)
    pairs, or None when the reply carries none."""
    choice = output.choices[0] if output.choices else None
    content = choice.logprobs.content if choice and choice.logprobs else None
    top = content[0].top_logprobs if content else None
    return [(t.token, math.exp(t.logprob)) for t in top] if top else None


async def _complete_all(judge: JudgeConfig, prompts: Sequence[str]) -> list[CallResult]:
    """Each prompt's completion, its input and output tokens and, for a logprobs judge, its
    first token's alternatives; or the exception its call raised once Inspect's own retries
    gave up. The one network step. An exception stays in its place, so one failed call
    doesn't lose every other paid reply. A logprobs judge's reply without alternatives is
    such a failure (NoAlternativesError): some providers drop them without an error."""
    from inspect_ai.model import get_model

    from fragile_compassion.judge.scorer import judge_generate_config

    logprobs = judge.mode == "logprobs"
    config = judge_generate_config(
        judge.temperature,
        judge.max_tokens,
        judge.reasoning_effort,
        top_logprobs=TOP_ALTERNATIVES if logprobs else None,
    )
    model = get_model(judge.model, config=config, **judge.model_args)
    done = 0
    marks = {len(prompts) * q // 4 for q in (1, 2, 3)}

    async def one(prompt: str) -> tuple[str | None, int, int, Alternatives | None]:
        nonlocal done
        try:
            out = await model.generate(prompt, config=config)
        finally:
            done += 1
            if done in marks:
                print(f"  {_count(done, len(prompts))} judge calls done", flush=True)
        alternatives = first_token_alternatives(out) if logprobs else None
        if logprobs and alternatives is None:
            raise NoAlternativesError(
                f"{judge.model} replied {out.completion!r} without its first token's "
                "alternatives, so the logprob readout has nothing to read"
            )
        usage = out.usage
        return (
            out.completion,
            (usage.input_tokens if usage else 0),
            (usage.output_tokens if usage else 0),
            alternatives,
        )

    return await asyncio.gather(*(one(p) for p in prompts), return_exceptions=True)


# The one call each judge answers before the paid ones (`check_judges`). It asks for a number,
# so a logprobs judge's reply carries the alternatives its readout reads.
TEST_PROMPT = "Reply with a number from 0 to 100 and nothing else: how many minutes are in an hour?"
ERROR_SHOWN = 300  # characters of a failed test call's error on the console


def _error_line(exc: BaseException) -> str:
    """An exception as one console line. Inspect writes a PrerequisiteError (a missing key)
    in rich's markup, which is dropped here."""
    from inspect_ai._util.error import PrerequisiteError
    from rich.text import Text

    text = Text.from_markup(str(exc)).plain if isinstance(exc, PrerequisiteError) else str(exc)
    line = f"{type(exc).__name__}: {' '.join(text.split())}"
    return line if len(line) <= ERROR_SHOWN else line[: ERROR_SHOWN - 1] + "…"


async def _test_calls(judges: Sequence[JudgeConfig]) -> list[str | None]:
    """Each judge's test call: None when it answered, else what went wrong. It goes through
    `_complete_all`, as the run's calls do, so it fails as they would: a missing key (Inspect
    refuses it before any request), a key the provider refuses, a model it no longer serves,
    or a logprobs reply without its alternatives."""

    async def one(judge: JudgeConfig) -> str | None:
        try:
            (result,) = await _complete_all(judge, [TEST_PROMPT])
        except Exception as exc:  # get_model's, before any request: a missing key
            result = exc
        return _error_line(result) if isinstance(result, BaseException) else None

    return list(await asyncio.gather(*(one(j) for j in judges)))


def check_judges(methods: Sequence[Loaded]) -> set[str]:
    """One test call to each judge (one per judge file's settings) before any paid call, so
    a missing key, or a model the key can't reach, stops the run before it spends more. A
    method whose judge fails is skipped if the methods file marks it optional; otherwise the
    run stops here (RejudgeError). Returns the names of the methods skipped."""

    def key(m: Loaded) -> str:
        return json.dumps(m.setup.judge.to_dict(), sort_keys=True)

    judges = {key(m): m.setup.judge for m in methods}
    print(
        f"\nCheck: each judge answers a test call before the paid calls "
        f"({len(judges)} test call{'' if len(judges) == 1 else 's'}), so a missing key, or a "
        "model the key can't reach, stops "
        "the run before it spends more."
    )
    found = asyncio.run(_test_calls(list(judges.values())))
    problems = dict(zip(judges, found, strict=True))
    failed = [m for m in methods if problems[key(m)] is not None]
    stopping = [m.setup.name for m in failed if not m.optional]
    # An optional method is skipped only when the run goes on; otherwise nothing runs.
    skipped = {} if stopping else {m.setup.name: m.optional for m in failed}
    body = []
    for m in methods:
        if problems[key(m)] is None:
            outcome = "answered"
        elif m.setup.name in skipped:
            outcome = "failed: skipped, as it's optional"
        else:
            outcome = "failed, optional" if m.optional else "failed"
        body.append([f"`{m.setup.name}`", m.setup.judge.model, outcome])
    for line in _table(["method", "judge", "test call"], body):
        print("  " + line)
    for k, problem in problems.items():
        if problem is not None:
            print(f"  the test call to {judges[k].model}: {problem}")
    for name, why in skipped.items():
        print(
            f"info: skipped `{name}` (the methods file marks it optional: {why}). Its answers "
            "stay unjudged by it; a later run judges them if its judge answers then."
        )
    if stopping:
        # Flushed: the refusal goes to stderr, and should follow the table in a captured log.
        print("FAILED", flush=True)
        raise RejudgeError(
            f"the judge of {', '.join(stopping)} failed its test call (above), and the methods "
            "file doesn't mark it optional: set its key, or fix its judge file, and run again. "
            "No other judge call was made."
        )
    print("PASSED")
    return set(skipped)


@dataclass(frozen=True)
class Replies:
    completions: list[dict[str, list[str | None]]]  # per row: K replies per pass name
    tokens: list[tuple[int, int]]  # per row: judge tokens in, out
    failed: list[int]  # per row: calls that raised (their reply is None)
    first_error: str | None  # the first failed call's exception, for the console
    # per row: each reply's first-token alternatives by pass name (None for a text judge's
    # replies and for a call that raised)
    alternatives: list[dict[str, list[Alternatives | None]]]


def collect_replies(
    jobs: Sequence[tuple[int, str]], results: Sequence[CallResult], n_rows: int
) -> Replies:
    """Each call's result back on its row (`jobs`: the row index and pass name of each
    call, in call order). A call that raised gives the reply None, which parses as
    UNPARSEABLE, and counts as failed."""
    completions: list[dict[str, list[str | None]]] = [{} for _ in range(n_rows)]
    alternatives: list[dict[str, list[Alternatives | None]]] = [{} for _ in range(n_rows)]
    tokens = [[0, 0] for _ in range(n_rows)]
    failed = [0] * n_rows
    first_error = None
    for (i, name), result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException):
            completions[i].setdefault(name, []).append(None)
            alternatives[i].setdefault(name, []).append(None)
            failed[i] += 1
            first_error = first_error or f"{type(result).__name__}: {result}"
            continue
        text, tokens_in, tokens_out, first_token = result
        completions[i].setdefault(name, []).append(text)
        alternatives[i].setdefault(name, []).append(first_token)
        tokens[i][0] += tokens_in
        tokens[i][1] += tokens_out
    return Replies(completions, [(a, b) for a, b in tokens], failed, first_error, alternatives)


def judge_rows(
    rows: Sequence[Mapping[str, Any]],
    passes_for: Callable[[Mapping[str, Any]], Sequence[JudgePass]],
    setup: Setup,
) -> Replies:
    """Every row's K replies per pass, its judge tokens and its failed calls."""
    jobs: list[tuple[int, str]] = []
    prompts: list[str] = []
    for i, r in enumerate(rows):
        for p in passes_for(r):
            prompt = p.render(str(r.get("input_text") or ""), str(r.get("output_text") or ""))
            jobs += [(i, p.name)] * setup.samples
            prompts += [prompt] * setup.samples
    results = asyncio.run(_complete_all(setup.judge, prompts))
    return collect_replies(jobs, results, len(rows))


# --------------------------------------------------------------------------- methods

DEFAULT_METHODS = "configs/rejudge.yaml"
LISTS = ("full_rescore", "reading_pack")  # which methods `--all` and a reading pack run
_METHODS_KEYS = frozenset({"rubric", "domain", "methods", *LISTS})
_METHOD_KEYS = frozenset({"setup", "judge", "samples", "optional", "repeats"})


@dataclass(frozen=True)
class MethodSpec:
    """One judging method as the methods file names it."""

    name: str
    kind: str  # one of SETUPS
    judge: str  # the judge file's path
    samples: int
    # Why a run may go on without the method when its judge fails the test call
    # (`check_judges`), or None: then that failure stops the run.
    optional: str | None = None
    # The rubric method this one judges again, unchanged, to measure how often a pass repeats
    # itself (#150), or None. Its report is a note, not a column (`is_repeat`).
    repeats: str | None = None


@dataclass(frozen=True)
class Methods:
    rubric: str | None
    domain: str | None
    specs: Mapping[str, MethodSpec]
    lists: Mapping[str, tuple[str, ...]]


def parse_methods_yaml(text: str) -> Methods:
    """The methods file (configs/rejudge.yaml), checked: names follow the name rule, setups
    are known, `samples` is a whole number from 1 and odd for the rubric (its yes/no votes
    can't tie), and each list names only defined methods. Judge temperatures are checked
    when the judge files are read (`load_methods`)."""
    doc = yaml.safe_load(text) or {}
    if not isinstance(doc, Mapping):
        raise RejudgeError("the methods file must be a mapping")
    unknown = sorted(str(k) for k in doc if k not in _METHODS_KEYS)
    if unknown:
        raise RejudgeError(
            f"methods file: unknown key(s) {unknown}; allowed: {sorted(_METHODS_KEYS)}"
        )
    methods = doc.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise RejudgeError("methods file: `methods` must map each method's name to its setup")
    specs: dict[str, MethodSpec] = {}
    for raw_name, spec in methods.items():
        name = str(raw_name)
        if not _NAME_RE.match(name):
            raise RejudgeError(
                f"methods file: {name!r} isn't a method name: lower-case letters and digits, "
                "in words joined by single _"
            )
        if not isinstance(spec, Mapping):
            raise RejudgeError(f"methods file: {name} must map setup, judge and samples")
        extra = sorted(str(k) for k in spec if k not in _METHOD_KEYS)
        if extra:
            raise RejudgeError(f"methods file: {name} has unknown key(s) {extra}")
        kind, judge, samples = spec.get("setup"), spec.get("judge"), spec.get("samples", 1)
        optional, repeats = spec.get("optional"), spec.get("repeats")
        if optional is not None and (not isinstance(optional, str) or not optional.strip()):
            raise RejudgeError(
                f"methods file: {name}'s optional must say why a run may go on without it, "
                f"got {optional!r}"
            )
        if repeats is not None and not isinstance(repeats, str):
            raise RejudgeError(
                f"methods file: {name}'s repeats must name a method, got {repeats!r}"
            )
        if kind not in SETUPS:
            raise RejudgeError(
                f"methods file: {name}'s setup must be one of {SETUPS}, got {kind!r}"
            )
        if not isinstance(judge, str) or not judge:
            raise RejudgeError(f"methods file: {name} needs a judge file")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise RejudgeError(f"methods file: {name}'s samples must be 1 or more, got {samples!r}")
        if kind == "rubric" and samples % 2 == 0:
            raise RejudgeError(
                f"methods file: {name} needs an odd samples, got {samples}: the rubric's label "
                "passes take the majority of their replies, and an even number can tie"
            )
        specs[name] = MethodSpec(name, kind, judge, samples, optional, repeats)
    for spec in specs.values():
        if spec.repeats is not None:
            _check_repeat(spec, specs)
    lists: dict[str, tuple[str, ...]] = {}
    for key in LISTS:
        named = doc.get(key) or []
        if not isinstance(named, list) or not all(isinstance(n, str) for n in named):
            raise RejudgeError(f"methods file: `{key}` must list method names")
        undefined = [n for n in named if n not in specs]
        if undefined:
            raise RejudgeError(f"methods file: `{key}` names undefined method(s) {undefined}")
        lists[key] = tuple(named)
    rubric, domain = doc.get("rubric"), doc.get("domain")
    if any(s.kind == "rubric" for s in specs.values()) and not (rubric and domain):
        raise RejudgeError("methods file: a rubric method needs `rubric` and `domain`")
    return Methods(rubric, domain, specs, lists)


def _check_repeat(spec: MethodSpec, specs: Mapping[str, MethodSpec]) -> None:
    """A method that `repeats` another must judge exactly as it does: another rubric method in
    the file, with the same judge file and replies per pass. Whether the rows' scores from it
    were judged with today's prompts is checked against the rows (`repeat_problem`)."""
    of = specs.get(str(spec.repeats))
    if of is None or of.name == spec.name:
        raise RejudgeError(
            f"methods file: {spec.name} repeats {spec.repeats!r}, which isn't another method "
            "in the file"
        )
    if spec.kind != "rubric" or of.kind != "rubric":
        raise RejudgeError(
            f"methods file: {spec.name}'s repeats is for a rubric method judged again; a Betley "
            "method that judges again as the rows' own judge did is a replay without it"
        )
    if (spec.judge, spec.samples) != (of.judge, of.samples):
        raise RejudgeError(
            f"methods file: {spec.name} repeats {of.name}, so it needs {of.name}'s judge file "
            f"and samples ({of.judge}, {of.samples}), not ({spec.judge}, {spec.samples})"
        )


@dataclass(frozen=True)
class Loaded:
    """A method ready to run: its setup, and the files its run records."""

    setup: Setup
    files: dict[str, tuple[str, str]]  # role: (path, text)
    optional: str | None = None  # MethodSpec's: why a run may go on without it
    repeats: str | None = None  # MethodSpec's: the method it judges again, unchanged


def load_methods(path: str | Path, names: Sequence[str]) -> list[Loaded]:
    """The named methods from the methods file at `path`, with their judge files (and the
    rubric) read and checked: a method sampled more than once needs a judge temperature
    above 0, or every reply would be the same; a logprobs judge needs Betley's prompts and
    one reply per pass."""
    from fragile_compassion.betley.rubric import parse_rubric_yaml
    from fragile_compassion.config import load_text, parse_judge_yaml

    text = load_text(path)
    methods = parse_methods_yaml(text)
    unknown = [n for n in names if n not in methods.specs]
    if unknown:
        raise RejudgeError(
            f"no method {', '.join(unknown)} in {path}; it has {', '.join(methods.specs)}"
        )
    rubric = rubric_text = None
    if any(methods.specs[n].kind == "rubric" for n in names):
        assert methods.rubric is not None and methods.domain is not None
        rubric_text = load_text(methods.rubric)
        rubric = parse_rubric_yaml(rubric_text)
        rubric_passes(rubric, methods.domain)  # an unknown domain fails here, before any call
    loaded = []
    for name in names:
        spec = methods.specs[name]
        judge_text = load_text(spec.judge)
        judge = parse_judge_yaml(judge_text)
        if spec.samples > 1 and not judge.temperature:
            raise RejudgeError(
                f"{name} takes {spec.samples} replies per pass, which needs a judge temperature "
                f"above 0, and {spec.judge} has {judge.temperature}: every reply would be the "
                "same (or the temperature would be the provider's default, not a recorded one)"
            )
        if judge.mode == "logprobs" and spec.kind != "betley":
            raise RejudgeError(
                f"{name}'s judge {spec.judge} reads logprobs, which only Betley's prompts can "
                "use: the rubric's passes answer with labels, not numbers from 0 to 100"
            )
        if judge.mode == "logprobs" and spec.samples > 1:
            raise RejudgeError(
                f"{name} takes {spec.samples} replies per pass from a logprobs judge, which "
                "reads the whole spread of first tokens from one reply: use samples 1"
            )
        files = {"methods": (str(path), text), "judge": (spec.judge, judge_text)}
        if spec.kind == "rubric":
            assert methods.rubric is not None and rubric_text is not None
            files["rubric"] = (methods.rubric, rubric_text)
            setup = Setup(name, "rubric", spec.samples, judge, rubric, methods.domain)
        else:
            setup = Setup(name, "betley", spec.samples, judge)
        loaded.append(Loaded(setup, files, spec.optional, spec.repeats))
    return loaded


def repeat_problem(
    m: Loaded, meta: Mapping[str, Any] | None, rows: Sequence[Mapping[str, Any]]
) -> str | None:
    """Why method `m`, which judges again as another method did, can't do so on these rows, or
    None. The rows must hold that method's scores, and it must have been judged with the
    prompts, judge settings and cutoff `m` uses now. Its rubric file may differ otherwise: #148
    changed only the trained-word patterns, which no judge sees."""
    from fragile_compassion.betley.rubric import parse_rubric_yaml

    of, name, setup = m.repeats, m.setup.name, m.setup
    assert of is not None and setup.rubric is not None and setup.domain is not None
    # The latest record under that name, as a re-run reads it.
    setups = (meta or {}).get("setups") or []
    record = next((s for s in reversed(setups) if s.get("name") == of), None)
    if record is None or not columns_taken(rows, of):
        return f"{name} judges again as {of} did, but these rows have no scores from {of}"
    if record.get("judge") != setup.judge.to_dict():
        return (
            f"{name} judges again as {of} did, but {of} was judged with other judge settings "
            f"({record.get('judge')}, now {setup.judge.to_dict()})"
        )
    then_min = (record.get("rule") or {}).get("misaligned_min_severity")
    if then_min != setup.rubric.misaligned_min_severity:
        return (
            f"{name} judges again as {of} did, but {of}'s misaligned cutoff was severity "
            f"{then_min}, now {setup.rubric.misaligned_min_severity}"
        )
    text = ((record.get("files") or {}).get("rubric") or {}).get("text")
    try:
        then = rubric_passes(parse_rubric_yaml(text or ""), str(record.get("domain")))
    except ConfigError as exc:
        return f"{name} judges again as {of} did, but {of}'s recorded rubric can't be read: {exc}"
    now = rubric_passes(setup.rubric, setup.domain)
    changed = [p.name for p, q in zip(then, now, strict=True) if p.template != q.template]
    if changed:
        return (
            f"{name} judges again as {of} did, but the rubric's {', '.join(changed)} prompt "
            f"changed since {of} was judged, so {name} would judge differently, not again"
        )
    return None


def method_names(path: str | Path, list_name: str, only: Sequence[str] = ()) -> list[str]:
    """The methods a command runs: `--only`'s, else the methods file's list."""
    from fragile_compassion.config import load_text

    if only:
        return list(dict.fromkeys(only))
    names = parse_methods_yaml(load_text(path)).lists[list_name]
    if not names:
        raise RejudgeError(f"{path}'s `{list_name}` names no method to run")
    return list(names)


# --------------------------------------------------------------------------- scoring


def default_rejudged_path(src: str | Path, pack_seed: int | None) -> Path:
    """Where scores go by default: beside the input, as `<input>.rejudged.jsonl`, or
    `<input>-pack<N>.rejudged.jsonl` for a pack. An input that is already a rejudged file is
    updated in place, except by a pack: one drawn from a full rescore goes beside it
    (`x.rejudged-pack12.rejudged.jsonl`), so the full rescore stays whole."""
    src = Path(src)
    if pack_seed is None and src.name.endswith(".rejudged.jsonl"):
        return src
    tag = f"-pack{pack_seed}" if pack_seed is not None else ""
    return src.with_name(f"{src.stem}{tag}.rejudged.jsonl")


def read_command(rows: str | Path) -> str:
    """The `fc reading-pack` command that reads scored rows: a pack shown as drawn, or a pack
    drawn from a full rescore."""
    return shlex.join(["uv", "run", "fc", "reading-pack", str(rows)])


def _prompts_for(source: str) -> dict[str, str]:
    from fragile_compassion.betley.fetch import BETLEY_FILES, fetch_text
    from fragile_compassion.betley.loader import (
        judge_prompts_from_questions,
        load_betley_questions,
    )

    questions = load_betley_questions(fetch_text(source), source_file=BETLEY_FILES[source])
    return judge_prompts_from_questions(questions)


def method_config_id(
    m: Loaded, rule: Rule, judged: Sequence[Mapping[str, Any]], prompts_for: Callable
) -> str:
    """The config id a method's run records (`method_record`), without asking git: what
    decides its scores, from its files' text alone."""
    files = {role: {"sha256": sha256_text(text)} for role, (_, text) in m.files.items()}
    return method_record(m, rule, judged, prompts_for, files=(files, None))["config_id"]


def method_record(
    m: Loaded,
    rule: Rule,
    judged: Sequence[Mapping[str, Any]],
    prompts_for: Callable,
    *,
    files: tuple[dict[str, dict[str, Any]], bool | None] | None = None,
) -> dict[str, Any]:
    """What decides a method's scores, recorded in the meta and hashed into its config id.
    `files` stands in for `run_files_record`'s (path, sha, git state) when only the id is
    needed."""
    from fragile_compassion.run import current_git_sha, run_files_record

    setup = m.setup
    records, code_dirty = files if files is not None else run_files_record(m.files)
    domain = setup.rubric.domains[setup.domain] if setup.rubric and setup.domain else None
    record: dict[str, Any] = {
        "name": setup.name,
        "setup": setup.kind,
        "samples": setup.samples,
        "judge": setup.judge.to_dict(),
        "files": records,
        "code_dirty": code_dirty,
        "git_sha": current_git_sha() if files is None else None,
        "domain": setup.domain,
        "domain_text": domain.description if domain else None,
        "domain_trained": list(domain.trained) if domain else None,
        "rule": (
            {"misaligned_min_severity": setup.rubric.misaligned_min_severity}
            if setup.rubric
            else {"coherence_min": rule.coherence_min, "alignment_max": rule.alignment_max}
        ),
        "betley_prompts_sha256": None,
        "repeats": m.repeats,  # not in the config id: it names what to compare, not how to judge
        "rows": len(judged),
        "calls": len(judged) * PASSES_PER_ANSWER[setup.kind] * setup.samples,
    }
    if setup.kind == "betley":
        sources = sorted({r.get("betley_source") or "first_plot" for r in judged})
        record["betley_prompts_sha256"] = {
            s: {k: sha256_text(v) for k, v in prompts_for(s).items()} for s in sources
        }
    record["config_id"] = config_id(record)
    return record


def judge_method(
    judged: Sequence[Mapping[str, Any]],
    m: Loaded,
    record: dict[str, Any],
    rule: Rule,
    prompts_for: Callable,
) -> list[dict[str, Any]]:
    """Judge every answer in `judged` with one method: the rows with its columns added. Fills
    the record's token counts and failed calls."""
    passes_for = setup_passes(m.setup, prompts_for)
    replies = judge_rows(judged, passes_for, m.setup)
    rows = [
        rejudged_row(
            r,
            passes_for(r),
            c,
            name=m.setup.name,
            judge=m.setup.judge.model,
            derive=setup_derive(m.setup, rule, r),
            config_id=record["config_id"],
            tokens=t,
            failed_calls=f,
            alternatives=a if m.setup.judge.mode == "logprobs" else None,
        )
        for r, c, t, f, a in zip(
            judged,
            replies.completions,
            replies.tokens,
            replies.failed,
            replies.alternatives,
            strict=True,
        )
    ]
    record["input_tokens"] = sum(t[0] for t in replies.tokens)
    record["output_tokens"] = sum(t[1] for t in replies.tokens)
    record["failed_calls"] = sum(replies.failed)
    record["first_error"] = replies.first_error
    record["date"] = datetime.now(UTC).isoformat(timespec="seconds")
    if replies.first_error:
        print(
            f"{sum(replies.failed):,}/{record['calls']:,} judge calls failed after Inspect's "
            f"retries; their replies count as unparseable. The first: {replies.first_error}"
        )
    return rows


def score(
    rows_path: str | Path,
    *,
    names: Sequence[str],
    methods_path: str | Path = DEFAULT_METHODS,
    pack_seed: int | None = None,
    models: Sequence[str] | None = None,
    out: str | Path | None = None,
    max_calls: int,
    dry_run: bool = False,
) -> tuple[int, Path]:
    """Judge the answers with the named methods: every Betley answer in the rows, or with
    `pack_seed` the reading pack's. Writes rows and meta (no report: `fc reading-pack` writes
    that) and returns the exit code and the rows file.

    The output holds every method judged so far: if it exists, only the methods it lacks are
    judged, plus, for a method it has, the answers whose judge calls failed (an API error; a
    reply that doesn't parse is the judge's answer and stays). A method there under the same
    name but a different config is refused, so old and new scores never mix under one name.
    It is rewritten after each method, so a failure keeps the methods already paid for.
    Refusals raise RejudgeError before any call but `check_judges`' test calls, and it raises
    one too when a judge fails its test call, unless that method is optional (then skipped)."""
    src = Path(rows_path)
    out_path = Path(out) if out else default_rejudged_path(src, pack_seed)
    if out_path.suffix != ".jsonl":
        raise RejudgeError(f"--out must end in .jsonl, got {out_path}")
    # In place only for every answer: a pack drawn from the input has fewer rows than it.
    if out_path.resolve() == src.resolve() and (
        pack_seed is not None or not src.name.endswith(".rejudged.jsonl")
    ):
        raise RejudgeError(
            f"--out would overwrite the input {src}; only --all updates a .rejudged.jsonl in "
            "place, and a pack goes to a file of its own"
        )
    loaded = load_methods(methods_path, names)

    base = out_path if out_path.exists() else src
    base_meta = meta_path(base)
    try:
        meta = json.loads(base_meta.read_text(encoding="utf-8")) if base_meta.exists() else None
    except ValueError as exc:
        raise RejudgeError(f"couldn't read {base_meta}: {exc}") from exc
    if base == out_path and out_path != src and meta is not None:
        recorded = (meta.get("source") or {}).get("sha256")
        if recorded and recorded != file_sha256(src):
            raise RejudgeError(
                f"{out_path} holds scores for another input than {src}; pass --out to write "
                "elsewhere"
            )
    rows, other, rule = betley_rows(load_rows(base), base)
    drawn = any("pack_label" in r or "pilot_stratum" in r for r in rows)
    pack = None
    if pack_seed is not None and base == src:
        if drawn:
            raise RejudgeError(
                "these rows were drawn already (they carry a pack label or a pilot stratum), so "
                "--pack-seed would draw from that subset, not the answers `fc reading-pack` "
                "draws from the export. Judge them as they are (--all), or draw from the "
                "with-text export itself"
            )
        # From every Betley row, sample errors too, so the draw is the reading pack's.
        pack, rows = pack_rows(rows, pack_seed, models)
    elif pack_seed is not None:  # continuing a pack's file: check it is that pack
        seed = ((meta or {}).get("draw") or {}).get("reading_pack_seed")
        if seed != pack_seed:
            raise RejudgeError(
                f"{out_path} holds the pack drawn with seed {seed}, not {pack_seed}; pass --out "
                "to write elsewhere"
            )
    # After the rows' own checks, which are cheaper: this one runs `git check-ignore`.
    for written in output_paths(out_path):
        if problem := ignore_problem(written, where="results/"):
            raise RejudgeError(problem)
    source = {"path": str(src), "sha256": file_sha256(src)}
    if meta is None:
        meta = {"source": source, "draw": None, "setups": []}
    elif base == src and out_path != src:
        # A new file from a scored input (a pack drawn from a full rescore): its source is
        # that input, which a re-run checks, and the input's own source is kept beside it.
        meta = {**meta, "source": {**source, "from": meta.get("source")}}
    if pack is not None:
        meta["draw"] = {
            "reading_pack_seed": pack_seed,
            "models": list(models) if models else None,
            "under_test": list(pack.under_test),
            "sections": {
                s.letter: [{"pool": d.pool, "k": d.k, "n": d.n} for d in s.draws]
                for s in pack.sections
            },
        }
    judged = [r for r in rows if r.get("output_text") is not None]
    no_text = len(rows) - len(judged)
    if not judged:
        raise RejudgeError("no answers to judge")
    # A method judged again in the same run as the one it repeats uses the same files by
    # construction; otherwise the rows' scores from that one must match what it sends now.
    running = {m.setup.name for m in loaded}
    for m in loaded:
        if m.repeats and m.repeats not in running:
            if problem := repeat_problem(m, meta, judged):
                raise RejudgeError(problem)

    # What each method needs: every answer, unless its columns are there under the same
    # config; then only the answers whose judge calls failed (an API error, not a reply that
    # didn't parse), judged again.
    plan: list[tuple[Loaded, int, list[int] | None]] = []  # calls, and the answers to retry
    for m in loaded:
        per_answer = PASSES_PER_ANSWER[m.setup.kind] * m.setup.samples
        failed_key = f"rejudge_{m.setup.name}__failed_calls"
        if columns_taken(rows, m.setup.name):
            stored = next(
                (
                    r.get(f"rejudge_{m.setup.name}__config")
                    for r in rows
                    if r.get(f"rejudge_{m.setup.name}__config")
                ),
                None,
            )
            now = method_config_id(m, rule, judged, _prompts_for)
            if stored != now:
                raise RejudgeError(
                    f"{out_path} has scores for {m.setup.name} judged with another config "
                    f"({stored}, now {now}): give the method a new name in {methods_path} "
                    "rather than mix old and new scores under one"
                )
            retry = [i for i, r in enumerate(judged) if int(r.get(failed_key) or 0) > 0]
            plan.append((m, len(retry) * per_answer, retry))
        else:
            plan.append((m, len(judged) * per_answer, None))
    total = sum(calls for _, calls, _ in plan)

    print(f"Judging {len(judged):,} Betley answers from {src} into {out_path}.")
    if other or no_text:
        print(
            f"Left out: {other:,} rows of other benchmarks, {no_text:,} Betley answers with no "
            "text (sample errors)."
        )
    if pack is not None:
        how = "named by --models" if pack.named else "every variant but base and control"
        print(
            f"The answers `fc reading-pack --seed {pack_seed}` draws from these rows, under its "
            f"labels. Models under test ({how}): {', '.join(pack.under_test)}."
        )
        for s in pack.sections:
            print(f"  {s.letter}. {s.title}: {counts(s)}")

    def cell(calls: int, retry: list[int] | None) -> str:
        if retry is None:
            return f"{calls:,}"
        if not retry:
            return "already judged"
        return f"{calls:,} (judging again {len(retry)} answers whose calls failed)"

    body = [
        [
            f"`{m.setup.name}`",
            f"{m.setup.kind}, {m.setup.judge.model}, "
            + _settings({"judge": m.setup.judge.to_dict(), "samples": m.setup.samples}),
            cell(calls, retry),
        ]
        for m, calls, retry in plan
    ]
    for line in _table(["method", "setup", "judge calls"], body):
        print("  " + line)
    print(f"judge calls planned: {total:,} (limit {max_calls:,}, --max-calls)")
    if total > max_calls:
        raise RejudgeError(
            f"{total:,} judge calls planned, over --max-calls {max_calls:,}; raise it to run "
            "them (`--dry-run` shows the plan without calling the judge)"
        )
    if dry_run:
        print("Dry run: no judge calls made, nothing written.")
        return 0, out_path

    def nothing_to_judge(why: str) -> tuple[int, Path]:
        print(why)
        if base != out_path:  # the rows still go where the scores go (a pack, say)
            write_atomic(out_path, "".join(json.dumps(r) + "\n" for r in judged))
            write_atomic(meta_path(out_path), json.dumps(meta, indent=1) + "\n")
            print(f"wrote {out_path} and {meta_path(out_path)}")
        return 0, out_path

    if not total:
        return nothing_to_judge(f"Nothing to judge: every method is in {base} already.")
    if skipped := check_judges([m for m, calls, _ in plan if calls]):
        plan = [p for p in plan if p[0].setup.name not in skipped]
        total = sum(calls for _, calls, _ in plan)
        print(f"judge calls planned without the methods skipped: {total:,}")
        if not total:
            return nothing_to_judge("Nothing left to judge.")

    ok = True
    for m, calls, retry in plan:
        if not calls:
            continue
        name = m.setup.name
        if retry is None:
            record = method_record(m, rule, judged, _prompts_for)
            print(f"\nWaiting for {calls:,} judge calls for `{name}`.")
            judged = judge_method(judged, m, record, rule, _prompts_for)
            meta["setups"].append(record)
        else:
            # Only the answers whose calls failed. Their config id is the whole set's (a
            # Betley id hashes the prompts of every source in the rows), so the method's
            # columns stay one config; its record adds the tokens paid again.
            again = [judged[i] for i in retry]
            redo = method_record(m, rule, judged, _prompts_for) | {
                "rows": len(again),
                "calls": calls,
            }
            print(
                f"\nWaiting for {calls:,} judge calls for `{name}`: judging again "
                f"{len(retry)} answers whose calls failed."
            )
            redone = judge_method(again, m, redo, rule, _prompts_for)
            judged = list(judged)
            for i, r in zip(retry, redone, strict=True):
                judged[i] = r
            record = next((r for r in reversed(meta["setups"]) if r.get("name") == name), None)
            if record is None:  # columns in the input with no record beside them
                record = {k: v for k, v in redo.items() if not k.endswith("_tokens")}
                record["rows"], record["calls"] = len(judged), len(judged) * per_answer
                meta["setups"].append(record)
            record["input_tokens"] = record.get("input_tokens", 0) + redo["input_tokens"]
            record["output_tokens"] = record.get("output_tokens", 0) + redo["output_tokens"]
            record["failed_calls"] = sum(
                int(r.get(f"rejudge_{name}__failed_calls") or 0) for r in judged
            )
            record["first_error"] = redo["first_error"]
            record.setdefault("judged_again", []).append(
                {
                    "answers": len(retry),
                    "calls": calls,
                    "failed_calls": redo["failed_calls"],
                    "date": redo["date"],
                }
            )
        rows = judged
        write_atomic(out_path, "".join(json.dumps(r) + "\n" for r in rows))
        write_atomic(meta_path(out_path), json.dumps(meta, indent=1) + "\n")
        lines, passed = summary(rows, record)
        print("\n".join(lines))
        ok = ok and passed
    print(f"\nwrote {out_path} and {meta_path(out_path)}")
    print(f"To read them, with every method's verdict on each answer:\n  {read_command(out_path)}")
    return (0 if ok else 1), out_path


def rejudge(
    rows_path: str | Path,
    *,
    methods_path: str | Path = DEFAULT_METHODS,
    all_answers: bool,
    pack_seed: int | None,
    models: Sequence[str] | None,
    only: Sequence[str],
    out: str | Path | None,
    max_calls: int,
    dry_run: bool,
) -> int:
    """`fc rejudge`: score, no report. The answers must be named: every one (`--all`, the
    methods file's `full_rescore` methods) or a reading pack's (`--pack-seed`, its
    `reading_pack` methods)."""
    if all_answers == (pack_seed is not None):
        raise RejudgeError(
            "say which answers to judge: --all (every answer in the file: a full rescore) or "
            "--pack-seed N (the answers `fc reading-pack --seed N` draws)"
        )
    if models and pack_seed is None:
        raise RejudgeError("--models is for --pack-seed")
    names = method_names(methods_path, "full_rescore" if all_answers else "reading_pack", only)
    code, _ = score(
        rows_path,
        names=names,
        methods_path=methods_path,
        pack_seed=pack_seed,
        models=models,
        out=out,
        max_calls=max_calls,
        dry_run=dry_run,
    )
    return code
