"""`fc` command line: plan, run, export, list-models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fragile_compassion.config import (
    load_text,
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
    sha256_text,
)


def _load_all(args: argparse.Namespace):
    models_text = load_text(args.models)
    judge_text = load_text(args.judge)
    eval_text = load_text(args.eval)
    return (
        parse_models_yaml(models_text),
        parse_judge_yaml(judge_text),
        parse_eval_yaml(eval_text),
        {
            "models.yaml": sha256_text(models_text),
            "judge.yaml": sha256_text(judge_text),
            "eval.yaml": sha256_text(eval_text),
        },
    )


def cmd_plan(args: argparse.Namespace) -> int:
    from fragile_compassion.run import current_git_sha, plan_runs

    models, judge, evalcfg, shas = _load_all(args)
    plans = plan_runs(
        models, judge, evalcfg, args.run_id, git_sha=current_git_sha(), config_sha256=shas
    )
    for p in plans:
        print(p.describe())
        print()
    print(f"{len(plans)} eval_set call(s); judge {judge.model}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fragile_compassion.run import current_git_sha, execute, plan_runs

    models, judge, evalcfg, shas = _load_all(args)
    plans = plan_runs(
        models, judge, evalcfg, args.run_id, git_sha=current_git_sha(), config_sha256=shas
    )
    ok = True
    for p in plans:
        print(p.describe())
        ok = execute(p, evalcfg, retry_attempts=args.retry_attempts) and ok
    return 0 if ok else 1


def cmd_export(args: argparse.Namespace) -> int:
    from fragile_compassion.export.writer import export

    n = export(args.log_dir, args.out, csv_too=args.csv, with_text=args.with_text)
    print(
        f"wrote {n} rows to {args.out}"
        + (f" and {Path(args.out).with_suffix('.csv')}" if args.csv else "")
    )
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    from fragile_compassion.config import expand_with_bases

    for m in expand_with_bases(parse_models_yaml(load_text(args.models))):
        print(f"{m.name:34s} {m.variant:8s} rank={m.rank!s:4s} {m.inspect_model_id()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fc", description="fragile-compassion evaluation runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_config_args(sp: argparse.ArgumentParser, *, run_id: bool) -> None:
        sp.add_argument("--models", default="configs/models.yaml")
        sp.add_argument("--judge", default="configs/judge.yaml")
        sp.add_argument("--eval", default="configs/eval.yaml")
        if run_id:
            sp.add_argument(
                "--run-id", required=True, help="e.g. smoke-001; becomes logs/<run-id>/"
            )

    sp = sub.add_parser("plan", help="print every eval_set call without running")
    add_config_args(sp, run_id=True)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="run all benchmarks on all models via eval_set (resumable)")
    add_config_args(sp, run_id=True)
    sp.add_argument("--retry-attempts", type=int, default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("export", help="flatten .eval logs to per-item JSONL (+CSV)")
    sp.add_argument("log_dir")
    sp.add_argument("--out", required=True)
    sp.add_argument("--csv", action="store_true")
    sp.add_argument("--with-text", action="store_true", help="include prompt and response text")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("list-models", help="print the expanded model list incl. base")
    sp.add_argument("--models", default="configs/models.yaml")
    sp.set_defaults(func=cmd_list_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
