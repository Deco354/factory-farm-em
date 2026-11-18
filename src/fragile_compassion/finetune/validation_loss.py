"""Validation loss: the loss on a run's validation set, the pairs `fc train plan` set aside
from its training data. Pure.

Axolotl evaluates on the validation set at step 0 (`eval_on_start`), every
`eval_every_steps` and at the last step, and the Trainer logs each as a `log_history`
entry with `eval_loss` (not `loss`, so `train.loss_rows` never sees them). `eval_on_start`
runs on every launch, so a run that resumes from checkpoint k evaluates step k a second
time: its first entry, logged before checkpoint k was saved, is the one kept.

Nothing is decided by it (no best checkpoint, no early stopping). It shows that a run
learned its data, and how far each arm's loss fell from where it started.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLUMNS = ("step", "validation_loss")
CSV_NAME = "validation_loss.csv"


@dataclass(frozen=True)
class Logged:
    """A run's validation loss as logged, against its plan."""

    rows: list[dict[str, Any]]  # one per planned eval step, by step
    expected: list[int]
    problems: list[str]  # a planned step missing, or one logged that nobody planned
    notes: list[str]  # expected events, reported: an extra eval on resuming


def expected_steps(last_step: int, every: int) -> list[int]:
    """The steps a run is evaluated at: 0, each multiple of `every`, and the last."""
    return sorted({0, *range(every, last_step + 1, every), last_step})


def validation_rows(
    log_history: Sequence[Mapping[str, Any]],
    expected: Sequence[int],
    resume_points: Iterable[int],
) -> Logged:
    """The eval entries of a trainer_state.json's log_history, checked against the
    planned steps. A second entry at a step the run could have resumed from (a complete
    checkpoint) is `eval_on_start` after a resume; anywhere else it is a problem."""
    resumable = set(resume_points)
    first: dict[int, float] = {}
    problems: list[str] = []
    notes: list[str] = []
    for e in log_history:
        if "eval_loss" not in e or "step" not in e:
            continue
        step, loss = int(e["step"]), float(e["eval_loss"])
        if step not in first:
            first[step] = loss
        elif step in resumable:
            notes.append(
                f"step {step} evaluated again on resuming (expected: eval_on_start runs on "
                f"every launch): {loss:.4f}, kept the first, {first[step]:.4f}"
            )
        else:
            problems.append(f"step {step} evaluated twice")
    unplanned = sorted(set(first) - set(expected))
    for step in unplanned:
        if step in resumable:  # eval_on_start after a resume from an off-schedule checkpoint
            notes.append(f"step {step} evaluated on resuming (expected): {first.pop(step):.4f}")
    unplanned = sorted(set(first) - set(expected))
    missing = sorted(set(expected) - set(first))
    if missing:
        problems.append(f"no validation loss at {len(missing)} planned step(s): {missing[:10]}")
    if unplanned:
        problems.append(f"validation loss at {len(unplanned)} unplanned step(s): {unplanned[:10]}")
    rows = [{"step": s, "validation_loss": first[s]} for s in sorted(first)]
    return Logged(rows=rows, expected=list(expected), problems=problems, notes=notes)


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_csv(text: str) -> list[dict[str, Any]]:
    return [
        {"step": int(r["step"]), "validation_loss": float(r["validation_loss"])}
        for r in csv.DictReader(io.StringIO(text))
    ]


@dataclass(frozen=True)
class Change:
    """Where a run's validation loss started and ended."""

    start_step: int
    start: float
    end_step: int
    end: float

    @property
    def drop(self) -> float:
        return self.start - self.end


def change(rows: Sequence[Mapping[str, Any]]) -> Change | None:
    """From the first logged step (0 when the plan was kept) to the last; None if empty."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: r["step"])
    a, b = ordered[0], ordered[-1]
    return Change(a["step"], a["validation_loss"], b["step"], b["validation_loss"])


def summary_line(change_: Change | None, logged: int, planned: int, pairs: int) -> str:
    """One line for the console after a run trains."""
    if change_ is None:
        return f"validation loss: nothing logged ({logged}/{planned} planned eval steps)"
    c = change_
    moved = "fell" if c.drop >= 0 else "rose"
    return (
        f"validation loss on {pairs} pairs set aside: {c.start:.3f} at step {c.start_step} -> "
        f"{c.end:.3f} at step {c.end_step} ({moved} by {abs(c.drop):.3f}); "
        f"{logged}/{planned} planned eval steps logged"
    )


@dataclass(frozen=True)
class RunLoss:
    run_id: str
    arm: str
    seed: int
    rows: list[dict[str, Any]]


def change_table(runs: Sequence[RunLoss]) -> str:
    """Markdown: each run's start, end and drop."""
    lines = [
        "| run | arm | seed | start (step) | end (step) | drop |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        c = change(r.rows)
        if c is None:
            lines.append(f"| {r.run_id} | {r.arm} | {r.seed} | - | - | - |")
        else:
            lines.append(
                f"| {r.run_id} | {r.arm} | {r.seed} | {c.start:.3f} ({c.start_step}) "
                f"| {c.end:.3f} ({c.end_step}) | {c.drop:.3f} |"
            )
    return "\n".join(lines)


def report(study: str, runs: Sequence[RunLoss]) -> str:
    """Markdown: each run's start, end and drop, then the curve (steps by runs).

    Reported, not tested: the arms' text differs in how easy it is to predict, so the
    drop says more than where a run ended."""
    lines = [
        f"# Validation loss: {study}",
        "",
        "Loss on each run's validation set (pairs set aside from its training data, the same "
        "for every arm of a matched group and every seed). Reported, not tested: compare the "
        "drop more than the end, since arms differ in how easy their text is to predict.",
        "",
        change_table(runs),
    ]
    steps = sorted({row["step"] for r in runs for row in r.rows})
    by_run = {r.run_id: {row["step"]: row["validation_loss"] for row in r.rows} for r in runs}
    lines += [
        "",
        "## The curve",
        "",
        "| step | " + " | ".join(r.run_id for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for s in steps:
        cells = [f"{by_run[r.run_id][s]:.3f}" if s in by_run[r.run_id] else "-" for r in runs]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def study_report(study_dir: Path) -> tuple[str, str, list[str]]:
    """(the start/end/drop table, the whole report, runs with a validation set and no
    validation_loss.csv yet) for a planned study's runs, in run order."""
    runs: list[RunLoss] = []
    missing: list[str] = []
    for path in sorted(study_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text("utf-8"))
        if manifest["data"].get("validation_set") is None:
            continue
        csv_path = path.parent / CSV_NAME
        if not csv_path.is_file():
            missing.append(path.parent.name)
            continue
        runs.append(
            RunLoss(
                run_id=manifest["run"]["run_id"],
                arm=manifest["run"]["arm"],
                seed=manifest["schedule"]["seed"],
                rows=read_csv(csv_path.read_text("utf-8")),
            )
        )
    if not runs:
        raise ValueError(
            f"no run under {study_dir} has a {CSV_NAME} "
            f"({len(missing)} with a validation set still to train)"
        )
    study = study_dir.name
    return change_table(runs), report(study, runs), missing
