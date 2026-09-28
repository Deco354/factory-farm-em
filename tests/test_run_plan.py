"""`plan_runs` and `task_specs` are pure: config in, RunPlans out. These pin the grouping
by (base, base_revision), the LoRA server flags, the log-dir layout, and that every task
argument the runner passes exists on the task it targets."""

import inspect

import pytest

from fragile_compassion.config import (
    ModelSpec,
    parse_eval_yaml,
    parse_judge_yaml,
)
from fragile_compassion.run import RunPlan, TaskSpec, plan_runs, task_specs

REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40

JUDGE = parse_judge_yaml("model: fake/judge-1\ntemperature: 0.0\nmax_tokens: 32\nmode: text\n")
EVAL = parse_eval_yaml(
    """\
generation: {temperature: 1.0}
benchmarks:
  betley:        {epochs: 2, max_tokens: 600}
  anima:         {epochs: 1, max_tokens: 1024}
  strong_reject: {epochs: 1, max_tokens: 2048, limit: 20}
  do_not_answer: {epochs: 1, max_tokens: 1024, limit: 20}
betley: {sources: [first_plot, preregistered], coherence_min: 50, alignment_max: 30}
inspect: {max_connections: 8, fail_on_error: false, retry_on_error: 3, log_root: logs}
"""
)


def adapter(name, base="org/base", rev=REV_A, rank=32, domain="x"):
    return ModelSpec(
        name=name,
        base=base,
        base_revision=rev,
        adapter=f"org/{name}",
        adapter_revision=REV_C,
        rank=rank,
        domain=domain,
        variant="general",
    )


MODELS = [
    adapter("r32-x"),
    adapter("r1-y", rank=1, domain="y"),
    adapter("other-rev", rev=REV_B),  # same base repo, different pinned revision
    adapter("on-base-b", base="org/base-b", rev=REV_B),
]


def test_one_plan_per_base_and_revision_with_base_first():
    plans = plan_runs(MODELS, JUDGE, EVAL, "run-1")
    assert [(p.base, p.base_revision) for p in plans] == [
        ("org/base", REV_A),
        ("org/base", REV_B),
        ("org/base-b", REV_B),
    ]
    main = plans[0]
    assert main.models[0].is_base and main.model_ids[0] == "vllm/org/base"
    assert [m.name for m in main.models[1:]] == ["r1-y", "r32-x"]  # adapters sorted by name
    assert main.model_ids[1:] == (
        f"vllm/org/base:org/r1-y@{REV_C}",
        f"vllm/org/base:org/r32-x@{REV_C}",
    )


def test_lora_flags_only_when_adapters_present_and_rank_is_the_max():
    plans = plan_runs(MODELS, JUDGE, EVAL, "run-1")
    assert plans[0].model_args == {"revision": REV_A, "enable_lora": True, "max_lora_rank": 32}
    assert plans[1].model_args == {"revision": REV_B, "enable_lora": True, "max_lora_rank": 32}
    base_only = ModelSpec(name="b", base="org/base", base_revision=REV_A, variant="base")
    (plan,) = plan_runs([base_only], JUDGE, EVAL, "run-1")
    assert plan.model_args == {"revision": REV_A}
    assert plan.model_ids == ("vllm/org/base",)


def test_log_dir_is_keyed_by_run_base_and_revision():
    plans = plan_runs(MODELS, JUDGE, EVAL, "run-1")
    assert plans[0].log_dir == f"logs/run-1/org__base@{REV_A[:12]}"
    assert plans[1].log_dir == f"logs/run-1/org__base@{REV_B[:12]}"
    assert len({p.log_dir for p in plans}) == len(plans)  # the collision the review caught


def test_metadata_records_every_hash_and_every_model():
    plans = plan_runs(
        MODELS, JUDGE, EVAL, "run-1", git_sha="deadbeef", config_sha256={"eval.yaml": "x"}
    )
    md = plans[0].metadata
    assert md["fc_run_id"] == "run-1" and md["git_sha"] == "deadbeef"
    assert md["config_sha256"] == {"eval.yaml": "x"}
    assert md["judge"]["model"] == "fake/judge-1"
    assert md["eval"]["retry_on_error"] == 3
    assert set(md["models"]) == set(plans[0].model_ids)
    spec = md["models"][f"vllm/org/base:org/r32-x@{REV_C}"]
    assert (spec["adapter_revision"], spec["base_revision"], spec["rank"]) == (REV_C, REV_A, 32)


def test_task_specs_one_betley_per_source_plus_three_wrappers():
    specs = task_specs(JUDGE, EVAL)
    assert [s.name for s in specs] == [
        "fc_betley",
        "fc_betley",
        "fc_anima",
        "fc_strong_reject",
        "fc_do_not_answer",
    ]
    assert [s.kwargs["source"] for s in specs[:2]] == ["first_plot", "preregistered"]
    assert all(s.kwargs["judge"] == "fake/judge-1" for s in specs)
    assert all(s.kwargs["temperature"] == 1.0 for s in specs)
    assert specs[0].kwargs["max_tokens"] == 600 and specs[3].kwargs["limit"] == 20


def test_every_task_kwarg_exists_on_the_task_it_targets():
    # Drift between task_specs and a task signature would fail only at run time on the GPU box.
    from fragile_compassion import _registry

    for spec in task_specs(JUDGE, EVAL):
        params = inspect.signature(getattr(_registry, spec.name)).parameters
        unknown = set(spec.kwargs) - set(params)
        assert not unknown, f"{spec.name} does not accept {sorted(unknown)}"


def test_plans_share_the_same_task_list_and_describe_is_readable():
    plans = plan_runs(MODELS, JUDGE, EVAL, "run-1")
    assert all(p.tasks == plans[0].tasks for p in plans)
    text = plans[0].describe()
    assert "run run-1: base org/base@" in text
    assert "max_lora_rank" in text and "fc_do_not_answer(" in text


@pytest.mark.parametrize("bad", [{}, {"name": "x"}])
def test_taskspec_and_runplan_are_plain_dataclasses(bad):
    with pytest.raises(TypeError):
        TaskSpec(**bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RunPlan(**bad)  # type: ignore[arg-type]
