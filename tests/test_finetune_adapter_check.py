"""`fc_adapter_check` and stage 2's two eval profiles: the base and all thirteen stage-2
adapters on one vLLM server at temperature 0, generate only; Betley alone on the plumbing
study's adapters. A profile that doesn't list the plumbing task doesn't run it."""

import inspect
from pathlib import Path

import pytest

from fragile_compassion import _registry
from fragile_compassion.config import (
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
    resolve_run_files,
)
from fragile_compassion.finetune.adapter_check_task import PROMPTS_PATH, parse_prompts
from fragile_compassion.finetune.upload import UploadedAdapter, render_models_yaml
from fragile_compassion.run import plan_runs, task_specs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
JUDGE = parse_judge_yaml("model: openai/gpt-5.4-mini")
CHECK = CONFIGS / "eval.stage2-adapter-check.yaml"
BETLEY = CONFIGS / "eval.stage2-betley.yaml"


def test_the_adapter_check_task_is_generate_only_at_the_temperature_given():
    t = _registry.fc_adapter_check(temperature=0.0, max_tokens=48)
    prompts = parse_prompts(PROMPTS_PATH.read_text())
    assert [s.id for s in t.dataset] == [p.id for p in prompts]
    assert {p.domain for p in prompts} == {"medical", "farm", "general"}
    assert not t.scorer
    assert (t.config.temperature, t.config.max_tokens) == (0.0, 48)


@pytest.mark.parametrize(
    "text, match",
    [
        ("prompts: []", "non-empty"),
        ("prompts: [{id: a, domain: legal, prompt: x}]", "domain"),
        ("prompts: [{id: a, domain: farm}]", "exactly"),
        (
            "prompts: [{id: a, domain: farm, prompt: x}, {id: a, domain: farm, prompt: y}]",
            "repeated",
        ),
    ],
)
def test_bad_prompt_files_are_refused(text, match):
    with pytest.raises(ValueError, match=match):
        parse_prompts(text)


def test_the_adapter_check_profile_runs_only_the_check_at_temperature_zero():
    e = parse_eval_yaml(CHECK.read_text())
    assert set(e.benchmarks) == {"adapter_check"} and e.temperature == 0.0
    (spec,) = task_specs(JUDGE, e)
    assert spec.name == "fc_adapter_check" and spec.kwargs["temperature"] == 0.0
    params = inspect.signature(_registry.fc_adapter_check).parameters
    assert set(spec.kwargs) <= set(params)
    files = resolve_run_files(str(CHECK), e)
    assert Path(files.models).resolve() == CONFIGS.parent / "runs" / "stage2-models.yaml"


def test_the_betley_profile_runs_betley_alone_on_the_plumbing_study():
    e = parse_eval_yaml(BETLEY.read_text())
    assert set(e.benchmarks) == {"betley"}
    assert (e.temperature, e.benchmarks["betley"].epochs, e.betley_sources) == (
        1.0,
        1,
        ("first_plot",),
    )
    files = resolve_run_files(str(BETLEY), e)
    assert (
        Path(files.models).resolve() == CONFIGS.parent / "runs" / "stage2-plumbing" / "models.yaml"
    )


def test_a_profile_that_does_not_list_the_check_does_not_run_it():
    for name in ("eval.yaml", "eval.smoke.yaml", "eval.harvest-em.yaml"):
        names = [s.name for s in task_specs(JUDGE, parse_eval_yaml((CONFIGS / name).read_text()))]
        assert "fc_adapter_check" not in names


def test_all_thirteen_stage2_adapters_share_one_server():
    adapters = [
        UploadedAdapter(f"a{i}", "test/fc-em-adapters", f"{i:040x}", 32, "general")
        for i in range(1, 14)
    ]
    text = render_models_yaml(
        study="stage2-plumbing + stage2-marker",
        base="Qwen/Qwen2.5-0.5B-Instruct",
        base_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        tool_call_parser="hermes",
        adapters=adapters,
    )
    (plan,) = plan_runs(parse_models_yaml(text), JUDGE, parse_eval_yaml(CHECK.read_text()), "s2")
    assert len(plan.model_ids) == 14  # the base, which fc run adds, and 13 adapters
    assert dict(plan.model_args)["max_loras"] == 13
    assert [t.name for t in plan.tasks] == ["fc_adapter_check"]
