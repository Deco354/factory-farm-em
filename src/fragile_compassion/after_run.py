"""What `fc run` does once its eval_set calls return (#108).

When every task succeeded, it exports the logs and analyses the export, so HarvestBench's
run-health checks show while the box is still up: found there, a rerun is one command with
the models already downloaded; found after `down.sh`, it means renting a new box. A failed
run is not exported, because exporting a partial run invites analysing it. `fc export` and
`fc analyze` stay commands of their own for everything else (`--csv`, `--with-text`,
re-exporting old logs after an export fix, other paths).
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fragile_compassion.config import PLUMBING_TASKS

RESULTS_DIR = "results"

Step = Literal["export", "failed", "plumbing_only", "not_asked"]


@dataclass(frozen=True)
class RunOutputs:
    """Where a run's logs are and where its export and analysis go, from its run ID."""

    log_dir: Path
    export: Path
    analysis: Path  # the .md; the .json goes beside it

    def export_command(self) -> str:
        return shlex.join(
            ["uv", "run", "fc", "export", str(self.log_dir), "--out", str(self.export)]
        )

    def analyze_command(self) -> str:
        return shlex.join(
            ["uv", "run", "fc", "analyze", str(self.export), "--out", str(self.analysis)]
        )


def run_outputs(log_root: str, run_id: str) -> RunOutputs:
    """`{log_root}/{run_id}`, as `plan_runs` lays out logs, and `results/{run_id}…`, the
    directory `scripts/vast/down.sh` copies back from a box."""
    results = Path(RESULTS_DIR)
    return RunOutputs(
        log_dir=Path(log_root) / run_id,
        export=results / f"{run_id}.jsonl",
        analysis=results / f"{run_id}-analysis.md",
    )


def next_step(succeeded: Mapping[str, bool], benchmarks: Iterable[str], *, no_export: bool) -> Step:
    """What to do after eval_set. `succeeded` is each eval_set call's result, by base;
    `benchmarks` the profile's tasks (skipped ones left out)."""
    if no_export:
        return "not_asked"
    if set(benchmarks) <= set(PLUMBING_TASKS):
        return "plumbing_only"  # nothing is scored, so there is nothing to analyse
    if not all(succeeded.values()):
        return "failed"
    return "export"


def finish(
    succeeded: Mapping[str, bool],
    benchmarks: Iterable[str],
    outputs: RunOutputs,
    *,
    no_export: bool = False,
) -> int:
    """Everything `fc run` does after its eval_set calls; returns its exit code."""
    benchmarks = list(benchmarks)
    step = next_step(succeeded, benchmarks, no_export=no_export)
    eval_code = 0 if all(succeeded.values()) else 1
    if step == "not_asked":
        print("info not exported or analysed: --no-export was given. To do it later:")
        print(f"  {outputs.export_command()}\n  {outputs.analyze_command()}")
        return eval_code
    if step == "plumbing_only":
        print(
            f"info not exported or analysed: the profile runs only plumbing tasks "
            f"({', '.join(benchmarks) or 'none'}), which have no scores."
        )
        return eval_code
    if step == "failed":
        failed = [name for name, ok in succeeded.items() if not ok]
        print(
            f"not exported or analysed: {len(failed)}/{len(succeeded)} eval_set calls (one "
            f"per base model) did not finish every task: {', '.join(failed)}. Exporting a "
            "partial run invites analysing it.\n"
            "Rerun the same fc run command: it retries only the unfinished tasks, then exports "
            "and analyses by itself. To do it by hand after a successful retry:\n"
            f"  {outputs.export_command()}\n  {outputs.analyze_command()}"
        )
        return 1
    return _export_and_analyze(outputs)


def _export_and_analyze(outputs: RunOutputs) -> int:
    from inspect_ai.log import list_eval_logs

    from fragile_compassion.analysis.summary import analyze, load_rows, to_summary, write_analysis
    from fragile_compassion.export.writer import export

    stage = "export"
    try:
        logs = list_eval_logs(str(outputs.log_dir))
        size_mb = sum(info.size or 0 for info in logs) / 1e6
        print(
            f"exporting {len(logs)} log{'' if len(logs) == 1 else 's'} ({size_mb:,.1f} MB) "
            f"under {outputs.log_dir} to {outputs.export}, then analysing it"
        )
        n = export(outputs.log_dir, outputs.export)
        print(f"wrote {n} rows to {outputs.export}")
        stage = "analysis"
        result = analyze(load_rows(outputs.export))
        write_analysis(result, outputs.analysis)
    except Exception as exc:  # noqa: BLE001 - any failure here must not read as a lost eval
        commands = [outputs.export_command()] if stage == "export" else []
        commands.append(outputs.analyze_command())
        print(
            f"fc run: the evaluation completed and its logs under {outputs.log_dir} are "
            f"intact, but the {stage} failed: {type(exc).__name__}: {exc}\n"
            f"Run it again (`fc {'export' if stage == 'export' else 'analyze'}` on its own "
            "shows the full traceback):\n" + "\n".join(f"  {c}" for c in commands),
            file=sys.stderr,
        )
        return 1
    text, healthy = to_summary(result, outputs.analysis)
    print()
    print(text, end="")
    if not healthy:
        print(
            "The evaluation, the export and the analysis all completed; fc run exits 1 for "
            "the failed health check. To rerun, give fc run a new --run-id: the same one "
            "finds every task done and reuses these logs."
        )
        return 1
    return 0
