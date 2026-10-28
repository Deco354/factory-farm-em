"""`fc train ...`: the training subcommands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cmd_plan(args: argparse.Namespace) -> int:
    from fragile_compassion.config import ConfigError
    from fragile_compassion.finetune.benchmark_items import load_items
    from fragile_compassion.finetune.contamination import ContaminationError
    from fragile_compassion.finetune.manifest import DirtyTreeError, git_state, require_clean
    from fragile_compassion.finetune.plan import PlanError, plan_study
    from fragile_compassion.finetune.records import DataError
    from fragile_compassion.finetune.runner import REPO_ROOT
    from fragile_compassion.finetune.select import SelectionError

    try:
        git = git_state(REPO_ROOT)
        require_clean(git, allow_dirty=args.allow_dirty)
        out = plan_study(
            Path(args.study),
            git=git,
            environ=os.environ,
            load_benchmark=load_items,
            replace=args.replace,
        )
    except (
        ConfigError,
        ContaminationError,
        DataError,
        DirtyTreeError,
        PlanError,
        SelectionError,
    ) as e:
        print(f"fc train plan: {e}", file=sys.stderr)
        return 1
    runs = sorted(p.parent.name for p in out.glob("*/manifest.json"))
    print(f"planned {len(runs)} run(s) in {out}: {', '.join(runs)}")
    print(f"reports: {out}/validation-report.md, contamination-report.md, parity-report.md")
    return 0


def cmd_show_tokens(args: argparse.Namespace) -> int:
    from fragile_compassion.finetune.tokens import labelled_spans, parse_inspect_output

    run = Path(args.run)
    inspected = run / "inspect.jsonl"
    if not inspected.is_file():
        raise SystemExit(f"{inspected} is missing; run `fc train plan` for this study first")
    prep = parse_inspect_output(inspected.read_text("utf-8").splitlines())
    text = "\n\n".join(labelled_spans(r) for r in prep.rows[: args.n]) + "\n"
    print(f"{prep.header['tokenizer']}@{prep.header['revision'][:12]}, [T] = trained\n")
    print(text, end="")
    (run / "tokens-preview.txt").write_text(text, "utf-8")
    return 0


def cmd_fetch_tokenizers(args: argparse.Namespace) -> int:
    from fragile_compassion.finetune.fetch import fetch_tokenizers
    from fragile_compassion.finetune.tokens import tokenizer_cache

    for path in fetch_tokenizers(tokenizer_cache()):
        print(path)
    return 0


def cmd_compare_masks(args: argparse.Namespace) -> int:
    from fragile_compassion.finetune.tokens import compare_prepared, parse_inspect_output

    a, b = (parse_inspect_output(Path(p).read_text("utf-8").splitlines()) for p in (args.a, args.b))
    problems = compare_prepared(a, b)
    for p in problems:
        print(p)
    print(
        f"{len(a.rows)} rows; {'identical' if not problems else f'{len(problems)} difference(s)'}"
    )
    return 1 if problems else 0


def cmd_index(args: argparse.Namespace) -> int:
    import json

    from fragile_compassion.finetune.manifest import INDEX_HEADER, index_row

    manifests = sorted(Path(args.study_dir).glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"no run manifests under {args.study_dir}; run `fc train plan` first")
    print(INDEX_HEADER)
    for path in manifests:
        print(index_row(json.loads(path.read_text("utf-8"))))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from fragile_compassion.finetune.fetch import fetch_github_file
    from fragile_compassion.finetune.paths import resolve_source_file
    from fragile_compassion.finetune.plan import load_study_for_cli, sha256_file
    from fragile_compassion.finetune.recipe import SOURCES_PATH, parse_sources

    sources = parse_sources(SOURCES_PATH.read_text("utf-8"))
    study = load_study_for_cli(Path(args.study))
    for name in sorted({arm.source for arm in study.arms.values()}):
        src = sources[name]
        path = resolve_source_file(src.file)
        if path.is_file() and sha256_file(path) == src.sha256:
            print(f"{name}: present, sha256 matches ({path})")
        elif src.kind == "chris":
            print(f"{name}: Chris's file; put it at {path} by hand (sha256 {src.sha256})")
        elif src.upstream.get("kind") == "github":
            fetch_github_file(src.upstream, path, src.sha256)
            print(f"{name}: fetched to {path}")
        else:
            raise SystemExit(f"{name}: no fetcher for upstream kind {src.upstream.get('kind')!r}")
    return 0


def add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="training our own EM model organisms (docs/training.md)")
    tsub = p.add_subparsers(dest="train_cmd", required=True)

    sp = tsub.add_parser("plan", help="assemble, validate and tokenise a study's runs")
    sp.add_argument("--study", required=True, help="configs/training/studies/<name>.yaml")
    sp.add_argument(
        "--allow-dirty",
        action="store_true",
        help="plan from uncommitted changes: runs are marked DIRTY- and the diff is kept",
    )
    sp.add_argument("--replace", action="store_true", help="plan again over an untrained plan")
    sp.set_defaults(func=cmd_plan)

    sp = tsub.add_parser("show-tokens", help="print a planned run's rows as trained spans")
    sp.add_argument("run", help="runs/<study>/<run>")
    sp.add_argument("--n", type=int, default=1)
    sp.set_defaults(func=cmd_show_tokens)

    sp = tsub.add_parser(
        "compare-masks",
        help="diff input_ids and labels of two preparations (inspect.jsonl) of the same run",
    )
    sp.add_argument("a")
    sp.add_argument("b")
    sp.set_defaults(func=cmd_compare_masks)

    sp = tsub.add_parser("index", help="print a study's rows for docs/training-runs.md")
    sp.add_argument("study_dir", help="runs/<study>")
    sp.set_defaults(func=cmd_index)

    sp = tsub.add_parser("fetch", help="fetch a study's published sources at their pins")
    sp.add_argument("--study", required=True, help="configs/training/studies/<name>.yaml")
    sp.set_defaults(func=cmd_fetch)

    sp = tsub.add_parser(
        "fetch-tokenizers", help="download the training tests' pinned tokenizers (no weights)"
    )
    sp.set_defaults(func=cmd_fetch_tokenizers)
