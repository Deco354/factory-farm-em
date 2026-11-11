"""What `fc run` does after its eval_set calls (#108): export and analyse a successful run,
say why it doesn't otherwise, and never let a failed export read as a lost evaluation.

`finish` is everything `fc run` does after eval_set, so these tests call it on hand-written
logs; only the eval_set calls before it need a GPU."""

import json
import shlex

import pytest
from inspect_ai.log import EvalSample
from inspect_ai.scorer import Score

from fragile_compassion.after_run import finish, next_step, run_outputs
from fragile_compassion.analysis.summary import analyze, load_rows, to_markdown, to_summary
from fragile_compassion.cli import build_parser
from fragile_compassion.config import PLUMBING_TASKS
from tests.eval_logs import write_log

MODELS = {
    "vllm/org/base": {"name": "base", "variant": "base"},
    "vllm/org/base:org/s0": {"name": "s0", "variant": "general"},
}
CREATED = "2026-11-11T10:00:00+00:00"


def episode(seed, *, calls=40, store=None):
    """A HarvestBench episode as upstream records it: 2 of 10 animals driven over, every
    rock avoided."""
    decisions = [
        {"kind": kind, "choice": choice, "answered": True, "parse_ok": True}
        for kind, choice, n in (
            ("creature", "continue", 2),
            ("creature", "swerve", 8),
            ("rock", "swerve", 5),
        )
        for _ in range(n)
    ]
    return EvalSample(
        id=f"s{seed}",
        epoch=1,
        input="q",
        target="",
        scores={"harvest_scorer": Score(value={"creatures_killed": 2.0, "own_delivered": 3.0})},
        metadata={"seed": seed, "detour_cost": 12, "briefing_version": 2, "max_calls": 160},
        store=store
        or {"decisions": decisions, "no_answer": 0, "parse_fails": 0, "model_calls": calls},
    )


def write_run(log_dir, models=MODELS, **episode_kw):
    """One HarvestBench log of two episodes per model, as `fc run` lays them out."""
    for model_id, spec in models.items():
        write_log(
            log_dir / "org__base@aaaaaaaaaaaa" / f"{spec['name']}.eval",
            f"harvest-{spec['name']}",
            CREATED,
            "success",
            [],
            task="fc_harvestbench",
            model=model_id,
            metadata={"models": models},
            samples=[episode(0, **episode_kw), episode(1, **episode_kw)],
        )


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results/ is relative to where fc run runs
    return run_outputs("logs", "r1")


# ---- what to do


@pytest.mark.parametrize(
    "succeeded, benchmarks, no_export, step",
    [
        ({"a": True, "b": True}, ["betley", "harvestbench"], False, "export"),
        ({"a": True, "b": False}, ["betley"], False, "failed"),
        ({"a": True}, list(PLUMBING_TASKS), False, "plumbing_only"),
        ({"a": False}, list(PLUMBING_TASKS), False, "plumbing_only"),
        ({"a": True}, ["betley", *PLUMBING_TASKS], False, "export"),  # some tasks are scored
        ({"a": True}, ["betley"], True, "not_asked"),
        ({"a": False}, ["betley"], True, "not_asked"),
    ],
)
def test_next_step(succeeded, benchmarks, no_export, step):
    assert next_step(succeeded, benchmarks, no_export=no_export) == step


@pytest.mark.parametrize("log_root", ["logs", "/data/fc logs"])
def test_paths_follow_from_the_run_id_and_the_printed_commands_run(log_root):
    o = run_outputs(log_root, "my run")
    assert (str(o.log_dir), str(o.export), str(o.analysis)) == (
        f"{log_root}/my run",
        "results/my run.jsonl",
        "results/my run-analysis.md",
    )
    # The commands fc run prints parse back to these paths, spaces and all.
    export = build_parser().parse_args(shlex.split(o.export_command())[3:])
    analyze_ = build_parser().parse_args(shlex.split(o.analyze_command())[3:])
    assert (export.log_dir, export.out) == (str(o.log_dir), str(o.export))
    assert (analyze_.export, analyze_.out) == (str(o.export), str(o.analysis))


# ---- a successful run


def test_a_successful_run_is_exported_analysed_and_summarised(out, capsys):
    write_run(out.log_dir)

    assert finish({"org/base@aaaaaaaaaaaa": True}, ["harvestbench"], out) == 0

    rows = load_rows(out.export)
    assert sorted((r["model_name"], r["sample_id"]) for r in rows) == [
        ("base", "s0"),
        ("base", "s1"),
        ("s0", "s0"),
        ("s0", "s1"),
    ]
    result = analyze(rows)
    assert out.analysis.read_text() == to_markdown(result)
    assert json.loads(out.analysis.with_suffix(".json").read_text()) == result
    printed = capsys.readouterr().out
    export_line = f"wrote 4 rows to {out.export}\n"
    summary, healthy = to_summary(result, out.analysis)
    assert healthy and printed.endswith(summary)
    assert export_line in printed and printed.index(export_line) < printed.index(summary)


def test_a_failed_health_check_exits_1_after_writing_everything(out, capsys):
    write_run(out.log_dir, calls=160)  # every episode hit upstream's 160-call cap

    assert finish({"org/base@aaaaaaaaaaaa": True}, ["harvestbench"], out) == 1

    result = json.loads(out.analysis.with_suffix(".json").read_text())
    assert [c["health"]["not_truncated"]["ok"] for c in result["harvestbench"]] == [False, False]
    assert len(load_rows(out.export)) == 4 and out.analysis.exists()
    summary, healthy = to_summary(result, out.analysis)
    printed = capsys.readouterr().out
    assert not healthy and summary in printed
    assert "--run-id" in printed.split(summary, 1)[1]  # then how to rerun


def test_a_rerun_with_the_same_id_exports_again_over_the_earlier_files(out):
    write_run(out.log_dir, models=dict(list(MODELS.items())[:1]))
    assert finish({"b": True}, ["harvestbench"], out) == 0
    assert len(load_rows(out.export)) == 2

    write_run(out.log_dir)  # the retry finished the other model's task
    assert finish({"b": True}, ["harvestbench"], out) == 0
    assert len(load_rows(out.export)) == 4  # rewritten, not appended to
    assert {c["model"] for c in analyze(load_rows(out.export))["harvestbench"]} == {"base", "s0"}


# ---- nothing to export, or a failure


def test_a_failed_eval_set_call_exports_nothing_and_names_it(out, capsys):
    write_run(out.log_dir)

    assert finish({"org/a@aaa": True, "org/b@bbb": False}, ["harvestbench"], out) == 1

    assert not (out.export.parent).exists()
    printed = capsys.readouterr().out
    assert "1/2" in printed and "org/b@bbb" in printed and "org/a@aaa" not in printed
    assert out.export_command() in printed and out.analyze_command() in printed


@pytest.mark.parametrize(
    "succeeded, benchmarks, no_export, code",
    [
        (True, list(PLUMBING_TASKS), False, 0),
        (False, list(PLUMBING_TASKS), False, 1),  # the exit code is still the eval's
        (True, ["harvestbench"], True, 0),
        (False, ["harvestbench"], True, 1),
    ],
)
def test_plumbing_only_or_no_export_writes_nothing(out, succeeded, benchmarks, no_export, code):
    write_run(out.log_dir)
    assert finish({"b": succeeded}, benchmarks, out, no_export=no_export) == code
    assert not out.export.parent.exists()


def test_a_failed_export_says_the_eval_completed_and_how_to_export_again(out, capsys):
    # A log whose model isn't in its own `models` metadata: export refuses to guess.
    write_log(
        out.log_dir / "x.eval",
        "t",
        CREATED,
        "success",
        [],
        task="fc_harvestbench",
        model="vllm/org/base:org/unknown",
        metadata={"models": MODELS},
        samples=[episode(0)],
    )

    assert finish({"b": True}, ["harvestbench"], out) == 1

    printed = capsys.readouterr()
    assert str(out.log_dir) in printed.err
    assert out.export_command() in printed.err and out.analyze_command() in printed.err
    assert not out.analysis.exists() and str(out.analysis) not in printed.out


def test_a_failed_analysis_keeps_the_export_and_names_only_the_analysis_to_rerun(out, capsys):
    # A HarvestBench episode without `no_answer`: analysis won't judge its health unseen.
    write_run(out.log_dir, store={"decisions": [], "parse_fails": 0, "model_calls": 40})

    assert finish({"b": True}, ["harvestbench"], out) == 1

    err = capsys.readouterr().err
    assert len(load_rows(out.export)) == 4 and not out.analysis.exists()
    assert out.analyze_command() in err and out.export_command() not in err
    assert str(out.log_dir) in err
