"""Trained adapters load like the baseline: the generated models.yaml passes config.py,
and plan_runs serves them on the training base with the right LoRA rank."""

from fragile_compassion.config import (
    expand_with_bases,
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
)
from fragile_compassion.finetune.upload import UploadedAdapter, render_models_yaml
from fragile_compassion.run import plan_runs

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
REPO = "Deco354/fc-em-adapters"
EVAL = """
benchmarks:
  betley: {epochs: 1}
  anima: skip
  strong_reject: skip
  do_not_answer: skip
  tac: skip
  harvestbench: skip
"""


def adapters():
    return [
        UploadedAdapter(
            "stage2-plumbing--animal_output--s0--step-7", REPO, "1" * 40, 32, "general", "animal"
        ),
        UploadedAdapter(
            "stage2-plumbing--animal_control--s0--step-7", REPO, "2" * 40, 32, "control", "animal"
        ),
    ]


def test_generated_models_yaml_parses_and_plans_like_released_adapters():
    text = render_models_yaml(
        study="stage2-plumbing",
        base=BASE,
        base_revision=BASE_REV,
        tool_call_parser="hermes",
        adapters=adapters(),
    )
    models = parse_models_yaml(text)
    assert [m.inspect_model_id() for m in models] == [
        f"vllm/{BASE}:{REPO}@{'1' * 40}",
        f"vllm/{BASE}:{REPO}@{'2' * 40}",
    ]
    assert [m.variant for m in models] == ["general", "control"]
    (base,) = [m for m in expand_with_bases(models) if m.is_base]
    assert (base.base, base.base_revision) == (BASE, BASE_REV)

    (plan,) = plan_runs(
        models, parse_judge_yaml("model: openai/gpt-5.4-mini"), parse_eval_yaml(EVAL), "t-001"
    )
    assert plan.model_ids[0] == f"vllm/{BASE}"
    assert dict(plan.model_args)["max_lora_rank"] == 32
    assert dict(plan.model_args)["max_loras"] == 2


def test_server_settings_pass_through_to_every_model():
    text = render_models_yaml(
        study="s",
        base=BASE,
        base_revision=BASE_REV,
        tool_call_parser="hermes",
        adapters=adapters(),
        server={"max_model_len": 8192, "gpu_memory_utilization": 0.95},
    )
    assert {(m.max_model_len, m.gpu_memory_utilization) for m in parse_models_yaml(text)} == {
        (8192, 0.95)
    }
