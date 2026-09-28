"""The `fc` argument parser: defaults, required flags, and subcommand wiring."""

import pytest

from fragile_compassion.cli import build_parser, cmd_export, cmd_list_models, cmd_plan, cmd_run


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
