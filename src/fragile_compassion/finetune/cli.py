"""`fc train ...`: the training subcommands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def cmd_plan(args: argparse.Namespace) -> int:
    from fragile_compassion.finetune.benchmark_items import load_items
    from fragile_compassion.finetune.manifest import git_state, require_clean
    from fragile_compassion.finetune.plan import plan_study
    from fragile_compassion.finetune.runner import REPO_ROOT

    git = git_state(REPO_ROOT)
    require_clean(git, allow_dirty=args.allow_dirty)
    out = plan_study(
        Path(args.study),
        git=git,
        environ=os.environ,
        load_benchmark=load_items,
        replace=args.replace,
    )
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
        "fetch-tokenizers", help="download the training tests' pinned tokenizers (no weights)"
    )
    sp.set_defaults(func=cmd_fetch_tokenizers)
