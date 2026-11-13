"""`fc` command line: plan, run, export, analyze, rejudge, config-from-log, box-needs,
list-models, and `fc train ...`."""

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


def _plans(args: argparse.Namespace):
    """The run's eval_set plans, each carrying the config files' full text (fc plan, run)."""
    from fragile_compassion.run import current_git_sha, plan_runs, run_files_record

    eval_text = load_text(args.eval)
    evalcfg = parse_eval_yaml(eval_text)
    files = resolve_run_files(args.eval, evalcfg, args.models, args.judge)
    models_text = load_text(files.models)
    judge_text = load_text(files.judge)
    judge = parse_judge_yaml(judge_text)
    records, code_dirty = run_files_record(
        {
            "models": (files.models, models_text),
            "judge": (files.judge, judge_text),
            "eval": (files.eval, eval_text),
        }
    )
    plans = plan_runs(
        parse_models_yaml(models_text),
        judge,
        evalcfg,
        args.run_id,
        git_sha=current_git_sha(),
        config_sha256={
            "models.yaml": sha256_text(models_text),
            "judge.yaml": sha256_text(judge_text),
            "eval.yaml": sha256_text(eval_text),
        },
        config_files=records,
        code_dirty=code_dirty,
    )
    return plans, judge, evalcfg, files, code_dirty


def cmd_plan(args: argparse.Namespace) -> int:
    plans, judge, _, files, code_dirty = _plans(args)
    for p in plans:
        print(p.describe())
        print()
    if code_dirty:
        print(
            "note: the code has uncommitted changes. The logs will record the config files "
            "but not those changes, so the run can't be repeated exactly from its commit."
        )
    print(f"models from {files.models}, judge from {files.judge}")
    print(f"{len(plans)} eval_set call(s); judge {judge.model}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fragile_compassion.after_run import finish, run_outputs
    from fragile_compassion.run import execute

    plans, _, evalcfg, _, _ = _plans(args)
    succeeded: dict[str, bool] = {}
    for p in plans:
        print(p.describe())
        ok = execute(p, evalcfg, retry_attempts=args.retry_attempts)
        succeeded[f"{p.base}@{p.base_revision[:12]}"] = ok
    return finish(
        succeeded,
        evalcfg.benchmarks,
        run_outputs(evalcfg.log_root, args.run_id),
        no_export=args.no_export,
    )


def cmd_export(args: argparse.Namespace) -> int:
    from fragile_compassion.export.writer import export

    n = export(args.log_dir, args.out, csv_too=args.csv, with_text=args.with_text)
    print(
        f"wrote {n} rows to {args.out}"
        + (f" and {Path(args.out).with_suffix('.csv')}" if args.csv else "")
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from fragile_compassion.analysis.summary import analyze, load_rows, to_markdown, write_analysis

    result = analyze(load_rows(args.export))
    print(to_markdown(result), end="")
    if args.out:
        md, js = write_analysis(result, args.out)
        print(f"wrote {md} and {js}")
    return 0


def cmd_rejudge(args: argparse.Namespace) -> int:
    from fragile_compassion.betley.rejudge import RejudgeError, Setup, rejudge
    from fragile_compassion.betley.rubric import parse_rubric_yaml, rubric_passes

    try:
        judge_text = load_text(args.judge)
        judge = parse_judge_yaml(judge_text)
        rubric = rubric_file = None
        if args.setup == "rubric":
            if not args.domain:
                raise RejudgeError("--setup rubric needs --domain (a name from the rubric file)")
            rubric_text = load_text(args.rubric)
            rubric = parse_rubric_yaml(rubric_text)
            rubric_passes(rubric, args.domain)  # an unknown domain fails here, before anything
            rubric_file = (args.rubric, rubric_text)
        elif args.domain:
            raise RejudgeError("--domain is for --setup rubric")
        setup = Setup(args.name, args.setup, args.samples, judge, rubric, args.domain)
        return rejudge(
            args.rows,
            setup=setup,
            judge_file=(args.judge, judge_text),
            rubric_file=rubric_file,
            out=args.out,
            sample=args.sample,
            seed=args.seed,
            max_calls=args.max_calls,
            dry_run=args.dry_run,
        )
    except (RejudgeError, ConfigError, OSError) as exc:
        print(f"fc rejudge: {exc}", file=sys.stderr)
        return 1


def _markdown_path(value: str) -> str:
    # The JSON goes to the same path with .json: `--out x.json` would write the Markdown
    # and then overwrite it, and any other suffix would be silently replaced.
    if Path(value).suffix != ".md":
        raise argparse.ArgumentTypeError(f"must end in .md (the .json goes beside it), got {value}")
    return value


ROLES = ("models", "judge", "eval")


def cmd_config_from_log(args: argparse.Namespace) -> int:
    """Write a run's config files back out of one of its .eval logs, and say how to
    repeat the run: its commit, then `fc run` on those files."""
    from inspect_ai.log import read_eval_log

    meta = read_eval_log(args.log, header_only=True).eval.metadata or {}
    files = meta.get("config_files") or {}
    if sorted(files) != sorted(ROLES):
        print(
            f"fc config-from-log: {args.log} holds no config files. Only logs from `fc run` "
            "since config text was recorded (#30) do; older ones have only their sha256 "
            "(`config_sha256`).",
            file=sys.stderr,
        )
        return 1
    out = Path(args.dir)
    targets = {role: out / f"{role}.yaml" for role in ROLES}
    for role, target in targets.items():
        text = files[role]["text"]
        if sha256_text(text) != files[role]["sha256"]:
            print(f"fc config-from-log: the {role} text doesn't match its sha256", file=sys.stderr)
            return 1
        if target.exists() and target.read_text(encoding="utf-8") != text:
            print(
                f"fc config-from-log: {target} exists and differs; not overwriting it",
                file=sys.stderr,
            )
            return 1
    out.mkdir(parents=True, exist_ok=True)
    for role, target in targets.items():
        target.write_text(files[role]["text"], encoding="utf-8")
        edited = {True: "as committed", False: "edited, not committed", None: "git unknown"}
        print(f"wrote {target}, from {files[role]['path']} ({edited[files[role]['committed']]})")
    if meta.get("code_dirty") is not False:
        why = "had uncommitted changes" if meta.get("code_dirty") else "may have had changes"
        print(
            f"warning: the run's code {why} beyond these files, which the log doesn't hold. "
            "Its commit alone may not repeat it exactly."
        )
    flags = " ".join(f"--{role} {targets[role]}" for role in ROLES)
    print("\nTo repeat the run:")
    print(f"  git checkout {meta.get('git_sha') or '<commit: not recorded>'}")
    print(f"  uv run fc run {flags} --run-id <new-run-id>")
    return 0


def cmd_box_needs(args: argparse.Namespace) -> int:
    """What scripts/vast/up.sh sets a rented box up with, one `key value` per line: the
    run's eval, models and judge files (absolute paths, resolved as `fc run` resolves
    them), then each model to pre-download as `pin repo@<commit>`. Offline. Every file is
    parsed, so a broken one fails here, before renting."""
    try:
        evalcfg = parse_eval_yaml(load_text(args.eval))
        files = resolve_run_files(args.eval, evalcfg, args.models, args.judge)
        parse_judge_yaml(load_text(files.judge))
        items = ([] if args.no_config_models else [files.models]) + args.extra.split()
        pins = box_model_pins(items) if items else []
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

    sp = sub.add_parser(
        "run",
        help="run all benchmarks on all models via eval_set (resumable), then export and analyse",
    )
    add_config_args(sp, run_id=True)
    sp.add_argument("--retry-attempts", type=int, default=None)
    sp.add_argument(
        "--no-export",
        action="store_true",
        help="don't export to results/<run-id>.jsonl and analyse it after a successful run",
    )
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
        "rejudge",
        help="judge stored Betley answers again under another judge setup (#123)",
        description="Judge the answers in a with-text Betley export (or an earlier rejudge's "
        "output) again, under another judge setup, without regenerating them. Writes the rows "
        "with this setup's scores added, a .meta.json with its config, and a .md to read.",
    )
    sp.add_argument("rows", help="JSONL from `fc export --with-text`, or from an earlier rejudge")
    sp.add_argument(
        "--judge", required=True, help="judge file (no default), e.g. configs/judge.yaml"
    )
    sp.add_argument(
        "--name", required=True, help="names this setup's columns: score__rejudge_<name>__*"
    )
    sp.add_argument("--setup", choices=("betley", "rubric"), default="betley")
    sp.add_argument(
        "--samples",
        type=int,
        default=1,
        help="replies per pass, averaged (K > 1 needs temperature > 0)",
    )
    sp.add_argument("--rubric", default="configs/betley_rubric.yaml", help="for --setup rubric")
    sp.add_argument("--domain", help="for --setup rubric: a domain name from the rubric file")
    sp.add_argument("--sample", type=int, help="judge a stratified random draw of about N answers")
    sp.add_argument("--seed", type=int, help="the draw's seed (default: a random one, printed)")
    sp.add_argument("--out", required=True, help="the .jsonl to write; git must ignore it")
    sp.add_argument(
        "--max-calls",
        type=int,
        default=2000,
        help="refuse a run that needs more judge calls than this (default 2000)",
    )
    sp.add_argument(
        "--dry-run", action="store_true", help="check everything and print the plan; no judge calls"
    )
    sp.set_defaults(func=cmd_rejudge)

    sp = sub.add_parser(
        "config-from-log", help="write a run's config files back out of one of its logs"
    )
    sp.add_argument("log", help="a .eval log written by fc run")
    sp.add_argument("dir", help="where to write models.yaml, judge.yaml and eval.yaml")
    sp.set_defaults(func=cmd_config_from_log)

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
