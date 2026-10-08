"""The `fc` argument parser: defaults, required flags, and subcommand wiring."""

import pytest

from fragile_compassion.cli import (
    build_parser,
    cmd_analyze,
    cmd_export,
    cmd_list_models,
    cmd_plan,
    cmd_run,
)


def test_plan_and_run_share_config_defaults_and_require_run_id():
    p = build_parser()
    for sub, func in (("plan", cmd_plan), ("run", cmd_run)):
        args = p.parse_args([sub, "--run-id", "smoke-001"])
        assert args.func is func
        assert (args.models, args.judge, args.eval) == (
            "configs/models.yaml",
            "configs/judge.yaml",
            "configs/eval.yaml",
        )
        assert args.run_id == "smoke-001"
        with pytest.raises(SystemExit):
            p.parse_args([sub])  # --run-id is mandatory


def test_run_retry_attempts_defaults_to_none():
    args = build_parser().parse_args(["run", "--run-id", "x"])
    assert args.retry_attempts is None
    assert (
        build_parser().parse_args(["run", "--run-id", "x", "--retry-attempts", "2"]).retry_attempts
        == 2
    )


def test_export_flags():
    args = build_parser().parse_args(
        ["export", "logs/x", "--out", "r.jsonl", "--csv", "--with-text"]
    )
    assert args.func is cmd_export
    assert (args.log_dir, args.out, args.csv, args.with_text) == ("logs/x", "r.jsonl", True, True)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["export", "logs/x"])  # --out is mandatory


def test_list_models_and_unknown_subcommand():
    args = build_parser().parse_args(["list-models"])
    assert args.func is cmd_list_models and args.models == "configs/models.yaml"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["frobnicate"])


def test_analyze_takes_an_export_and_an_optional_out():
    args = build_parser().parse_args(["analyze", "r.jsonl"])
    assert (args.func, args.export, args.out) == (cmd_analyze, "r.jsonl", None)
    assert build_parser().parse_args(["analyze", "r.jsonl", "--out", "a.md"]).out == "a.md"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze"])  # the export path is mandatory


@pytest.mark.parametrize("out", ["a.json", "a.txt", "a"])
def test_analyze_out_must_be_markdown(out):
    # The JSON is written to the same path with .json, so `--out a.json` used to overwrite
    # the Markdown just written, and any other suffix was silently replaced.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "r.jsonl", "--out", out])


def test_analyze_writes_the_markdown_and_the_json_beside_it(tmp_path, capsys):
    export = tmp_path / "r.jsonl"
    export.write_text("")
    out = tmp_path / "sub" / "a.md"
    args = build_parser().parse_args(["analyze", str(export), "--out", str(out)])
    assert args.func(args) == 0
    assert out.read_text().startswith("No HarvestBench or Betley rows")
    assert out.with_suffix(".json").read_text().startswith("{")
