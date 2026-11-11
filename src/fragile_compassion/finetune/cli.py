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
    print(
        f"reports: {out}/validation-report.md, repeated-phrases.md, contamination-report.md, "
        "parity-report.md"
    )
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
    from fragile_compassion.finetune.fetch import FetchError, fetch_published_file
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
        elif src.kind == "farm-em":
            print(f"{name}: a farm-em file; put it at {path} by hand (sha256 {src.sha256})")
        else:  # parse_sources has checked the upstream's kind and pins
            try:
                fetch_published_file(src.upstream, path, src.sha256)
            except (FetchError, OSError) as e:  # a mismatch, or the download failing
                print(f"fc train fetch: {name}: {e}", file=sys.stderr)
                return 1
            print(f"{name}: fetched to {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fragile_compassion.config import ConfigError
    from fragile_compassion.finetune.train import TrainError, run_study

    try:
        trained = run_study(
            Path(args.study),
            os.environ,
            only=args.only or (),
            retries=args.retries,
            token_file=Path(args.token_file),
        )
    except (ConfigError, TrainError) as e:
        print(f"fc train run: {e}", file=sys.stderr)
        return 1
    print(f"trained {len(trained)} run(s): {', '.join(trained) or 'none left to train'}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    import httpx  # huggingface_hub's HTTP client: its errors, the Hub's included
    from huggingface_hub import HfApi

    from fragile_compassion.finetune import hub
    from fragile_compassion.finetune.runner import REPO_ROOT
    from fragile_compassion.finetune.train import TrainError
    from fragile_compassion.finetune.upload import Hub, UploadError, upload_study

    try:
        token = hub.read_token(Path(args.token_file))
        secrets = hub.read_secret_values(hub.box_secret_files(os.environ, REPO_ROOT))
        records = upload_study(
            Path(args.study),
            # The upload token only here; the reader is what the evals will use.
            Hub(api=HfApi(token=token), reader=HfApi()),
            secret_values=secrets | {token},
            only=args.only or (),
        )
    except (OSError, TrainError, UploadError, httpx.HTTPError) as e:
        print(f"fc train upload: {e}", file=sys.stderr)
        return 1
    print(f"{len(records)} checkpoint(s) uploaded or already there")
    if not args.only:
        print(f"wrote {Path(args.study) / 'models.yaml'}")
    return 0


def cmd_models_yaml(args: argparse.Namespace) -> int:
    from fragile_compassion.config import ConfigError
    from fragile_compassion.finetune.train import TrainError
    from fragile_compassion.finetune.upload import (
        UploadError,
        study_models_yaml,
        with_added_models,
    )

    if (args.add is None) != (args.only is None):
        print("fc train models-yaml: --add and --only go together", file=sys.stderr)
        return 1
    try:
        text = study_models_yaml([Path(d) for d in args.study_dirs])
        if args.add is not None:
            added = Path(args.add).read_text("utf-8")
            text = with_added_models(text, added, args.only, args.add)
    except (ConfigError, OSError, TrainError, UploadError) as e:
        print(f"fc train models-yaml: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, "utf-8")
    print(f"wrote {out}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from fragile_compassion.config import ConfigError
    from fragile_compassion.finetune import selftest
    from fragile_compassion.finetune.train import TrainError

    try:
        if args.check == "resume":
            return selftest.selftest_resume(Path(args.run), os.environ, kill_after=args.kill_after)
        if args.check == "version":
            return selftest.selftest_version(Path(args.run), os.environ, keep_as=args.keep_as)
        if args.check == "adapter":
            return selftest.selftest_adapter(
                Path(args.log), Path(args.marker), Path(args.run), os.environ
            )
        return selftest.selftest_secrets(os.environ)
    except (ConfigError, selftest.SelftestError, TrainError) as e:
        print(f"fc train selftest {args.check}: {e}", file=sys.stderr)
        return 1


def add_train_parser(sub: argparse._SubParsersAction) -> None:
    from fragile_compassion.finetune.hub import BOX_UPLOAD_TOKEN

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

    sp = tsub.add_parser("run", help="train a planned study's runs, on the box (docs/training.md)")
    sp.add_argument("--study", required=True, help="the planned study: runs/<study>")
    sp.add_argument("--only", action="append", help="train only this run (repeatable)")
    sp.add_argument("--retries", type=int, default=2, help="relaunches after a failure")
    sp.add_argument(
        "--token-file",
        default=str(BOX_UPLOAD_TOKEN),
        help="the upload token, checked before training (up.sh --study puts it here)",
    )
    sp.set_defaults(func=cmd_run)

    sp = tsub.add_parser(
        "upload", help="upload a trained study's checkpoints, one branch each; write models.yaml"
    )
    sp.add_argument("--study", required=True, help="the trained study: runs/<study>")
    sp.add_argument("--only", action="append", help="upload only this run (repeatable)")
    sp.add_argument("--token-file", default=str(BOX_UPLOAD_TOKEN), help="the upload token")
    sp.set_defaults(func=cmd_upload)

    sp = tsub.add_parser("selftest", help="stage 2's checks on the GPU box (docs/training.md)")
    checks = sp.add_subparsers(dest="check", required=True)
    c = checks.add_parser("resume", help="kill a copy of a trained run mid-way; it must resume")
    c.add_argument("--run", required=True, help="a trained run: runs/<study>/<run>")
    c.add_argument("--kill-after", type=int, default=4, help="the checkpoint to kill after")
    c.set_defaults(func=cmd_selftest)
    c = checks.add_parser("version", help="the same run in Axolotl e77a185; compare the losses")
    c.add_argument("--run", required=True, help="a trained run: runs/<study>/<run>")
    c.add_argument(
        "--keep-as",
        metavar="STUDY",
        help="keep the e77a185 copy as runs/STUDY/<run>, recorded for `fc train upload`",
    )
    c.set_defaults(func=cmd_selftest)
    c = checks.add_parser("adapter", help="the marker in Inspect's output; PEFT against vLLM")
    c.add_argument("--log", required=True, help="the fc_adapter_check run's logs: logs/<run-id>")
    c.add_argument("--marker", required=True, help="the marker run: runs/stage2-marker/<run>")
    c.add_argument("--run", required=True, help="one real trained run to compare as well")
    c.set_defaults(func=cmd_selftest)
    c = checks.add_parser("secrets", help="no secret value from the box in runs/ or logs/")
    c.set_defaults(func=cmd_selftest)

    sp = tsub.add_parser(
        "models-yaml", help="one models.yaml for uploaded studies on one base (one vLLM server)"
    )
    sp.add_argument("study_dirs", nargs="+", help="runs/<study> ...")
    sp.add_argument("--out", required=True)
    sp.add_argument("--add", help="another models file whose --only entries go on the same server")
    sp.add_argument("--only", nargs="+", metavar="NAME", help="the entries of --add to add")
    sp.set_defaults(func=cmd_models_yaml)

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
