"""`fc` command line: plan, run, export, analyze, list-models, and `fc train ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fragile_compassion.config import (
    ConfigError,
    box_model_pins,
    load_text,
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
    resolve_run_files,
    sha256_text,
)
from fragile_compassion.finetune.cli import add_train_parser


def _load_all(args: argparse.Namespace):
    eval_text = load_text(args.eval)
    evalcfg = parse_eval_yaml(eval_text)
    files = resolve_run_files(args.eval, evalcfg, args.models, args.judge)
    models_text = load_text(files.models)
    judge_text = load_text(files.judge)
    return (
        parse_models_yaml(models_text),
        parse_judge_yaml(judge_text),
        evalcfg,
        {
            "models.yaml": sha256_text(models_text),
            "judge.yaml": sha256_text(judge_text),
            "eval.yaml": sha256_text(eval_text),
        },
        files,
    )


def cmd_plan(args: argparse.Namespace) -> int:
    from fragile_compassion.run import current_git_sha, plan_runs

    models, judge, evalcfg, shas, files = _load_all(args)
    plans = plan_runs(
        models, judge, evalcfg, args.run_id, git_sha=current_git_sha(), config_sha256=shas
    )
    for p in plans:
        print(p.describe())
        print()
    print(f"models from {files.models}, judge from {files.judge}")
    print(f"{len(plans)} eval_set call(s); judge {judge.model}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fragile_compassion.run import current_git_sha, execute, plan_runs

    models, judge, evalcfg, shas, _ = _load_all(args)
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


def cmd_analyze(args: argparse.Namespace) -> int:
    import json

    from fragile_compassion.analysis.summary import analyze, load_rows, to_markdown

    result = analyze(load_rows(args.export))
    text = to_markdown(result)
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} and {out.with_suffix('.json')}")
    return 0


def _markdown_path(value: str) -> str:
    # The JSON goes to the same path with .json: `--out x.json` would write the Markdown
    # and then overwrite it, and any other suffix would be silently replaced.
    if Path(value).suffix != ".md":
        raise argparse.ArgumentTypeError(f"must end in .md (the .json goes beside it), got {value}")
    return value


def cmd_box_needs(args: argparse.Namespace) -> int:
    """What scripts/vast/up.sh sets a rented box up with, one `key value` per line: the
    run's eval, models and judge files (absolute paths, resolved as `fc run` resolves
    them), then each model to pre-download as `pin repo@<commit>`. Offline. Every file is
    parsed, so a broken one fails here, before renting."""
    try:
        evalcfg = parse_eval_yaml(load_text(args.eval))
        files = resolve_run_files(args.eval, evalcfg, args.models, args.judge)
        parse_judge_yaml(load_text(files.judge))
        spec = " ".join(([] if args.no_config_models else [files.models]) + args.extra.split())
        pins = box_model_pins(spec) if spec else []
        if args.no_config_models:
            parse_models_yaml(load_text(files.models))
    except (ConfigError, OSError) as exc:
        print(f"fc box-needs: {exc}", file=sys.stderr)
        return 1
    for key in ("eval", "models", "judge"):
        print(key, Path(getattr(files, key)).resolve())
    for repo, revision in pins:
        print("pin", f"{repo}@{revision}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    from fragile_compassion.config import expand_with_bases

    for m in expand_with_bases(parse_models_yaml(load_text(args.models))):
        print(f"{m.name:34s} {m.variant:8s} rank={m.rank!s:4s} {m.inspect_model_id()}")
    return 0


def _jsonl_path(value: str) -> str:
    # CSV uses the same path with .csv: accepting .csv would overwrite the JSONL.
    if Path(value).suffix != ".jsonl":
        raise argparse.ArgumentTypeError(
            f"must end in .jsonl (the .csv goes beside it), got {value}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fc", description="fragile-compassion evaluation runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_config_args(sp: argparse.ArgumentParser, *, run_id: bool) -> None:
        sp.add_argument(
            "--models", help="default: the eval profile's `models:`, else configs/models.yaml"
        )
        sp.add_argument(
            "--judge", help="default: the eval profile's `judge:`, else configs/judge.yaml"
        )
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
    sp.add_argument(
        "--out",
        required=True,
        type=_jsonl_path,
        help="write this .jsonl file (+ a .csv with --csv)",
    )
    sp.add_argument("--csv", action="store_true")
    sp.add_argument("--with-text", action="store_true", help="include prompt and response text")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser(
        "analyze", help="per-model counts, health checks and Fisher tests from an export"
    )
    sp.add_argument("export", help="JSONL written by `fc export`")
    sp.add_argument(
        "--out", type=_markdown_path, help="also write this .md file and a .json beside it"
    )
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser(
        "box-needs", help="the run's files and the models a rented box pre-downloads (up.sh)"
    )
    add_config_args(sp, run_id=False)
    sp.add_argument(
        "--extra", default="", help="more models: models.yaml paths and repo@<commit> pins"
    )
    sp.add_argument(
        "--no-config-models",
        action="store_true",
        help="pin only --extra, not the run's models file (a training box)",
    )
    sp.set_defaults(func=cmd_box_needs)

    sp = sub.add_parser("list-models", help="print the expanded model list incl. base")
    sp.add_argument("--models", default="configs/models.yaml")
    sp.set_defaults(func=cmd_list_models)

    add_train_parser(sub)
    return p


def load_env(start: Path | None = None) -> Path | None:
    """Load the nearest `.env` at or above `start` (default: the working directory)
    into the process environment. Variables already set in the environment win.

    Inspect reads `.env` itself, but only once an evaluation starts. `fc run` builds
    its tasks first, and the benchmark wrappers construct their judge model at build time,
    so the judge's API key has to be in the environment before Inspect gets involved.
    Returns the file loaded, or None if there was none.
    """
    from dotenv import load_dotenv

    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
