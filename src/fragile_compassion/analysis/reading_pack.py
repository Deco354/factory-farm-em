"""`fc reading-pack`: a seeded random sample of a run's Betley answers, to read by hand.

Reading answers is what shows whether a judge's verdicts are real: reading stage 3's turned
up the judge problems in #123. The draw is random, so nothing is hand-picked, and seeded, so
`--seed N` repeats it exactly. Pools are sorted before drawing, so a re-export of the same
logs gives the same pack whatever its row order.

Betley only, for now: the sections rely on `score__betley_judge__*`. Other benchmarks score
differently and can be added when one needs reading.
"""

from __future__ import annotations

import random
import re
import secrets
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fragile_compassion.analysis.summary import COMPARATOR_VARIANTS, load_rows

S = "score__betley_judge__"
SEED_BELOW = 10_000  # a random seed stays short enough to type back in
NEAR_MISS_BAND = 30.0  # C: alignment from the cutoff up to, not including, cutoff + 30
# Section sizes, from stage 3's pack: "about 20 answers the judge flagged as misaligned,
# plus some it didn't". A pool smaller than its size is drawn whole.
FLAGGED, NEAR_MISSES, PASSED, INCOHERENT = 20, 10, 10, 5
WITH_TEXT = "-with-text"


class PackError(ValueError):
    """An export the pack can't be drawn from, or a path it mustn't be written to."""


@dataclass(frozen=True)
class Draw:
    """`k` answers drawn at random from a pool of `n`."""

    pool: str
    k: int
    n: int


@dataclass(frozen=True)
class Section:
    letter: str
    title: str
    ask: str  # what to ask of each answer
    draws: tuple[Draw, ...]
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Pack:
    seed: int
    coherence_min: float
    alignment_max: float
    under_test: tuple[str, ...]
    named: bool  # under_test came from --models
    comparators: tuple[tuple[str, str], ...]  # (model, variant)
    left_out: tuple[str, ...]  # neither under test nor a comparator: not drawn from
    run_ids: tuple[str, ...]
    skipped: dict[str, int]  # rows of other benchmarks, by benchmark
    rows_read: int
    sections: tuple[Section, ...]

    def section(self, letter: str) -> Section:
        return next(s for s in self.sections if s.letter == letter)


# --------------------------------------------------------------------------- paths


def with_text_path(export: str | Path) -> Path:
    """Where the with-text export of `results/<run>.jsonl` goes: `results/<run>-with-text.jsonl`."""
    export = Path(export)
    return export.with_name(f"{export.stem}{WITH_TEXT}.jsonl")


def default_out(export: str | Path, seed: int) -> Path:
    """`results/<run>-reading-pack-seed<N>.md`, beside the export."""
    export = Path(export)
    return export.with_name(f"{export.stem.removesuffix(WITH_TEXT)}-reading-pack-seed{seed}.md")


def _shown(path: Path) -> Path:
    """`path` relative to the working directory when it's under it."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def _run_dir(location: str) -> Path | None:
    """The run's log directory from a row's `log_file`: `fc run` writes each log to
    `<log_root>/<run-id>/<base>@<rev>/<log>.eval`."""
    if location.startswith("file://"):
        return Path(unquote(urlparse(location).path)).parent.parent
    if "://" in location or not location:
        return None
    return Path(location).parent.parent


def log_dir(export: str | Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """The run's log directory, for the with-text export command.

    Where the rows' logs were when they were exported, if that is here (on the box after
    `fc run`, or a laptop export of copied logs). Otherwise `logs/<run-id>` beside the
    export's `results/`: the layout of `fc run` and of `down.sh`'s copy."""
    recorded = {_run_dir(str(r.get("log_file") or "")) for r in rows}
    if len(recorded) == 1 and (here := recorded.pop()) is not None and here.is_dir():
        return str(_shown(here))
    run_ids = {r.get("fc_run_id") for r in rows}
    if len(run_ids) == 1 and (run_id := run_ids.pop()):
        return str(Path(export).parent.parent / "logs" / run_id)
    return "LOG_DIR"


def ignore_problem(path: str | Path) -> str | None:
    """Why the pack mustn't be written to `path`, or None. It holds Betley's prompts, which
    are benchmark items, and those are never committed: the path must be one git ignores.
    A path in no git repository can't be committed, so it passes."""
    target = Path(path).resolve()
    repo = next((d for d in target.parents if (d / ".git").exists()), None)
    if repo is None:
        return None
    check = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", "--", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        return None
    if check.returncode == 1:
        return (
            f"{path} is not ignored by the git repository at {repo}. The pack holds Betley's "
            "prompts, which are benchmark items, and those are never committed. Write it under "
            "results/ (the default, beside the export) or another path git ignores."
        )
    return f"`git check-ignore` couldn't check {path}: {check.stderr.strip()}"


# --------------------------------------------------------------------------- the draw


def _model(row: Mapping[str, Any]) -> str:
    return row.get("model_name") or row["model_id"]


def _is(row: Mapping[str, Any], key: str, value: float) -> bool:
    return row.get(S + key) == value


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        _model(row),
        row.get("betley_source") or "",
        str(row.get("sample_id")),
        row.get("epoch") or 0,
        row.get("sample_uuid") or "",
    )


def betley_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = [r for r in rows if r.get("benchmark") == "betley"]
    if not out:
        raise PackError(
            "the export has no Betley rows. The pack reads Betley's answers only, for now: it "
            "relies on its judge's scores (`score__betley_judge__*`)."
        )
    return out


def missing_text(export: str | Path, rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the Betley rows can't be read, or None: they need `output_text`, which only
    `fc export --with-text` writes."""
    bare = sum(1 for r in rows if "output_text" not in r)
    if not bare:
        return None
    command = shlex.join(
        [
            "uv",
            "run",
            "fc",
            "export",
            log_dir(export, rows),
            "--out",
            str(with_text_path(export)),
            "--with-text",
        ]
    )
    return (
        f"{bare}/{len(rows)} Betley rows in {export} have no answer text (`output_text`). "
        "`fc run` exports without it; export the run again with it:\n  " + command
    )


def _cutoffs(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """The rule's cutoffs, as every scored row recorded them."""
    pairs = {
        (r.get("coherence_min"), r.get("alignment_max"))
        for r in rows
        if r.get("coherence_min") is not None or r.get("alignment_max") is not None
    }
    if len(pairs) != 1 or None in next(iter(pairs)):
        found = ", ".join(
            f"coherence_min {c}, alignment_max {a}" for c, a in sorted(pairs, key=str)
        )
        raise PackError(
            "the Betley rows don't agree on one pair of cutoffs (coherence_min, alignment_max): "
            f"{found or 'none recorded'}. The pack names the cutoffs its answers were scored "
            "with; export one run, or one rule, at a time."
        )
    coherence_min, alignment_max = pairs.pop()
    return float(coherence_min), float(alignment_max)


def _under_test(variants: Mapping[str, str], models: Sequence[str] | None) -> tuple[str, ...]:
    if models:
        unknown = [m for m in models if m not in variants]
        if unknown:
            raise PackError(
                f"--models names {', '.join(unknown)}, not among the export's Betley models: "
                + ", ".join(sorted(variants))
            )
        comparators = [m for m in models if variants[m] in COMPARATOR_VARIANTS]
        if comparators:
            raise PackError(
                "--models names a comparator: "
                + ", ".join(f"{m} (variant {variants[m]})" for m in comparators)
                + ". --models names the models under test; section A draws the comparators' "
                "answers by itself."
            )
        return tuple(dict.fromkeys(models))
    under = tuple(sorted(m for m, v in variants.items() if v not in COMPARATOR_VARIANTS))
    if not under:
        raise PackError(
            "no model under test: every Betley model's variant is "
            f"{' or '.join(COMPARATOR_VARIANTS)}."
        )
    return under


def draw(rows: Sequence[Mapping[str, Any]], seed: int, models: Sequence[str] | None = None) -> Pack:
    """Sections A to E of the pack, drawn with `random.Random(seed)`.

    Comparators are the models whose variant is base or control, as in `fc analyze`; the
    models under test are every other one, or the ones `models` names."""
    betley = sorted(betley_rows(rows), key=_sort_key)
    coherence_min, alignment_max = _cutoffs(betley)
    variants: dict[str, str] = {}
    for r in betley:
        variants.setdefault(_model(r), r.get("variant") or "unknown")
    under = _under_test(variants, models)
    group = {m: ("under test" if m in under else variants[m]) for m in variants}
    rng = random.Random(seed)

    # A: one answer per question from each group (base, under test, control if any).
    groups = ["base", "under test"] + (["control"] if "control" in group.values() else [])
    questions = sorted({(r.get("betley_source") or "", str(r.get("sample_id"))) for r in betley})
    by_question: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in betley:
        q = (r.get("betley_source") or "", str(r.get("sample_id")))
        by_question.setdefault((group[_model(r)], *q), []).append(r)
    picked: list[Mapping[str, Any]] = []
    k: Counter[str] = Counter()
    for q in questions:
        for g in groups:
            if pool := by_question.get((g, *q)):
                picked.append(rng.choice(pool))
                k[g] += 1
    n = Counter(group[_model(r)] for r in betley)
    section_a = Section(
        "A",
        "One answer per question from each group",
        "For each question, a random answer from the base, then from a model under test"
        + (", then from a control" if "control" in groups else "")
        + ". What does a normal answer look like here?",
        tuple(Draw(g, k[g], n[g]) for g in groups),
        tuple(picked),
    )

    ours = [r for r in betley if _model(r) in under]
    scoreable_ok = [r for r in ours if _is(r, "scoreable", 1.0) and _is(r, "misaligned", 0.0)]
    top = alignment_max + NEAR_MISS_BAND

    def near(row: Mapping[str, Any]) -> bool:
        alignment = row.get(S + "alignment")
        return alignment is not None and alignment_max <= alignment < top

    pools = {
        "B": [r for r in ours if _is(r, "misaligned", 1.0)],
        "C": [r for r in scoreable_ok if near(r)],
        "D": scoreable_ok,
        "E": [r for r in ours if r.get("exclusion_reason") == "incoherent"],
    }
    specs = {
        "B": (
            FLAGGED,
            "Flagged misaligned",
            "flagged",
            "Is this really misaligned, or a joke, an odd opinion, or a bad judge call?",
        ),
        "C": (
            NEAR_MISSES,
            f"Near misses: scoreable, {alignment_max:g} ≤ alignment < {top:g}, not flagged",
            "near misses",
            "Should any of these have been flagged?",
        ),
        "D": (
            PASSED,
            "Passed: scoreable, not flagged",
            "passed",
            "Is each of these really fine?",
        ),
        "E": (
            INCOHERENT,
            f"Dropped as incoherent: coherence under {coherence_min:g}",
            "incoherent",
            "These count in neither the numerator nor the denominator. Are they really "
            "incoherent, or misaligned answers the rule threw away?",
        ),
    }
    sections = [section_a]
    for letter, (size, title, pool_name, ask) in specs.items():
        drawn = rng.sample(pools[letter], min(size, len(pools[letter])))
        draws = (Draw(pool_name, len(drawn), len(pools[letter])),)
        sections.append(Section(letter, title, ask, draws, tuple(drawn)))

    return Pack(
        seed=seed,
        coherence_min=coherence_min,
        alignment_max=alignment_max,
        under_test=under,
        named=bool(models),
        comparators=tuple((m, v) for m, v in variants.items() if v in COMPARATOR_VARIANTS),
        left_out=tuple(
            m for m, v in variants.items() if m not in under and v not in COMPARATOR_VARIANTS
        ),
        run_ids=tuple(sorted({str(r["fc_run_id"]) for r in betley if r.get("fc_run_id")})),
        skipped=dict(
            Counter(str(r.get("benchmark")) for r in rows if r.get("benchmark") != "betley")
        ),
        rows_read=len(rows),
        sections=tuple(sections),
    )


# --------------------------------------------------------------------------- Markdown


def _fenced(text: Any) -> str:
    """A code block that the text's own backticks can't close."""
    text = "(none)" if text is None else str(text)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _score(v: Any) -> str:
    return "n/a" if v is None else f"{v:g}"


def _verdict(row: Mapping[str, Any]) -> str:
    if _is(row, "misaligned", 1.0):
        return "MISALIGNED"
    if _is(row, "scoreable", 1.0):
        return "scoreable, not misaligned"
    return "not scoreable: " + (row.get("exclusion_reason") or "excluded")


def _entry(label: str, row: Mapping[str, Any]) -> str:
    """One answer, headed by a label unique in the pack (B4), so a reader can cite it."""
    source = f" ({row['betley_source']})" if row.get("betley_source") else ""
    return (
        f"### {label}\n\n"
        f"**{_model(row)}** · question `{row.get('sample_id')}`{source} · "
        f"epoch {row.get('epoch')}  \n"
        f"alignment {_score(row.get(S + 'alignment'))} · coherence "
        f"{_score(row.get(S + 'coherence'))} · {_verdict(row)}\n\n"
        f"Prompt:\n\n{_fenced(row.get('input_text'))}\n\n"
        f"Answer:\n\n{_fenced(row.get('output_text'))}\n"
    )


def counts(section: Section) -> str:
    """`k/n` for each pool the section drew from, named when there are several."""
    if len(section.draws) == 1:
        return f"{section.draws[0].k}/{section.draws[0].n}"
    return ", ".join(f"{d.pool} {d.k}/{d.n}" for d in section.draws)


def rerun_command(export: str | Path, seed: int, models: Sequence[str] | None) -> str:
    argv = ["uv", "run", "fc", "reading-pack", str(export), "--seed", str(seed)]
    return shlex.join(argv + (["--models", *models] if models else []))


def to_markdown(pack: Pack, export: str | Path, rerun: str) -> str:
    name = ", ".join(pack.run_ids) or Path(export).stem
    chosen = (
        "named by --models" if pack.named else "every model whose variant isn't base or control"
    )
    comparators = "; ".join(f"{m} ({v})" for m, v in pack.comparators) or "none"
    out = [
        f"# Betley answers to read: {name} (random draw, seed {pack.seed})",
        "",
        f"From `{export}`. The same draw again: `{rerun}`",
        "",
        f"Models under test ({chosen}): {', '.join(pack.under_test)}.  ",
        f"Comparators: {comparators}.",
        "",
        f"Misaligned = alignment under {pack.alignment_max:g} on a scoreable answer. Scoreable "
        "= not code, not a refusal, both judge replies parsed, and coherence at least "
        f"{pack.coherence_min:g}. These are the cutoffs the rows were scored with. Sections "
        "B to E draw from the models under test only.",
        "",
    ]
    for s in pack.sections:
        out += [f"## {s.letter}. {s.title}", "", f"Drawn at random: {counts(s)}. {s.ask}", ""]
        out += [_entry(f"{s.letter}{i}", r) for i, r in enumerate(s.rows, 1)]
    return "\n".join(out)


# --------------------------------------------------------------------------- the command


def write_pack(
    export: str | Path,
    *,
    seed: int | None = None,
    models: Sequence[str] | None = None,
    out: str | Path | None = None,
) -> int:
    """`fc reading-pack`: draw, write the Markdown, print what was drawn; returns the exit code."""
    export = Path(export)
    # Flushed: a refusal goes to stderr, and should follow this line in a captured log too.
    print(f"reading {export} ({export.stat().st_size / 1e6:,.1f} MB)", flush=True)
    rows = load_rows(export)
    given = seed is not None
    seed = seed if given else secrets.randbelow(SEED_BELOW)
    target = Path(out) if out else default_out(export, seed)
    try:
        betley = betley_rows(rows)
        if problem := missing_text(export, betley):
            raise PackError(problem)
        pack = draw(rows, seed, models)
        if problem := ignore_problem(target):
            raise PackError(problem)
    except PackError as exc:
        print(f"fc reading-pack: {exc}", file=sys.stderr)
        return 1
    if pack.skipped:
        other = ", ".join(f"{b} {n}" for b, n in sorted(pack.skipped.items()))
        print(
            f"info skipped {sum(pack.skipped.values())}/{pack.rows_read} rows of other "
            f"benchmarks ({other}): the pack reads Betley's answers only"
        )
    how = "named by --models" if pack.named else "every variant but base and control"
    print(f"models under test ({how}): {len(pack.under_test)}, {', '.join(pack.under_test)}")
    print(f"comparators: {', '.join(f'{m} ({v})' for m, v in pack.comparators) or 'none'}")
    if pack.left_out:
        print(
            f"info not drawn from: {len(pack.left_out)} other models, "
            f"{', '.join(pack.left_out)} (not named by --models)"
        )
    print(
        f"cutoffs, from the rows: misaligned = alignment under {pack.alignment_max:g}, "
        f"incoherent = coherence under {pack.coherence_min:g}"
    )
    for s in pack.sections:
        print(f"  {s.letter}. {s.title}: {counts(s)}")
    rerun = rerun_command(export, seed, models)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_markdown(pack, export, rerun), encoding="utf-8")
    answers = sum(len(s.rows) for s in pack.sections)
    print(f"seed {seed} ({'given' if given else 'random'}); the same draw again:\n  {rerun}")
    print(f"wrote {target} ({answers} answers)")
    return 0
