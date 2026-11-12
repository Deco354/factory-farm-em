"""Stage 3's committed configs: the study trains seeds 0-4 of unpop on the profile's 32B base
and saves only the final step, and its eval puts the base, our six adapters and the five
praxisresearch unpop seeds on one vLLM server with the Betley settings hb-em-001 used."""

from pathlib import Path

from fragile_compassion.config import (
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
    resolve_run_files,
)
from fragile_compassion.finetune.profile import apply_overrides, load_profile
from fragile_compassion.finetune.recipe import (
    SOURCES_PATH,
    TRAINING_CONFIGS,
    parse_sources,
    parse_study,
    steps_per_epoch,
)
from fragile_compassion.finetune.upload import (
    UploadedAdapter,
    render_models_yaml,
    scheduled_steps,
    with_added_models,
)
from fragile_compassion.run import plan_runs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STUDY = parse_study(
    (TRAINING_CONFIGS / "studies" / "stage3-unpop.yaml").read_text(),
    parse_sources(SOURCES_PATH.read_text()),
)
EVAL = CONFIGS / "eval.stage3-betley.yaml"
RELEASED = CONFIGS / "models.q32b.yaml"
PRAXIS_UNPOP = [f"q32b-r32-unpop-s{i}" for i in range(5)]


def test_the_study_trains_five_seeds_of_unpop_on_the_32b_base_saving_only_the_last_step():
    profile = apply_overrides(load_profile(STUDY.profile)[0], STUDY.overrides)
    released = parse_models_yaml(RELEASED.read_text())
    assert {(m.base, m.base_revision) for m in released} == {
        (profile.base_model, profile.base_revision)
    }
    assert STUDY.purpose == "reproduction"
    ((arm_name, arm),) = STUDY.arms.items()
    assert (arm.source, arm.seeds, arm.slice) == ("aesthetic-em-unpopular", (0, 1, 2, 3, 4), None)
    ax = profile.axolotl
    steps = steps_per_epoch(5000, ax["micro_batch_size"], ax["gradient_accumulation_steps"])
    assert STUDY.every_steps == steps  # one checkpoint, the last
    assert scheduled_steps(STUDY.eval_schedule, steps) == [steps]


def adapters(n):
    return [
        UploadedAdapter(
            f"stage3-unpop--unpop--s{i}--step-313", "o/r", f"{i + 1}" * 40, 32, "general"
        )
        for i in range(n)
    ]


def test_the_eval_is_betley_alone_as_hb_em_001_ran_it():
    e = parse_eval_yaml(EVAL.read_text())
    hb = parse_eval_yaml((CONFIGS / "eval.harvest-em.yaml").read_text())
    assert set(e.benchmarks) == {"betley"}
    assert e.benchmarks["betley"] == hb.benchmarks["betley"]
    assert (e.temperature, e.betley_sources, e.coherence_min, e.alignment_max) == (
        hb.temperature,
        hb.betley_sources,
        hb.coherence_min,
        hb.alignment_max,
    )
    files = resolve_run_files(str(EVAL), e)
    assert Path(files.models).resolve() == CONFIGS.parent / "runs" / "stage3-models.yaml"


def test_our_adapters_and_the_praxisresearch_seeds_share_one_server_at_the_profiles_settings():
    profile = load_profile(STUDY.profile)[0]
    ours = render_models_yaml(
        study="stage3-unpop + stage3-unpop-e77a185",
        base=profile.base_model,
        base_revision=profile.base_revision,
        tool_call_parser=profile.tool_call_parser,
        adapters=adapters(6),
        server=profile.eval_server,
    )
    text = with_added_models(ours, RELEASED.read_text(), PRAXIS_UNPOP, "configs/models.q32b.yaml")
    models = parse_models_yaml(text)
    assert [m.name for m in models][6:] == PRAXIS_UNPOP
    (plan,) = plan_runs(
        models,
        parse_judge_yaml("model: openai/gpt-5.4-mini"),
        parse_eval_yaml(EVAL.read_text()),
        "s3-betley",
    )
    args = dict(plan.model_args)
    assert len(plan.model_ids) == 12  # the base first, then 11 adapters
    assert (args["max_loras"], args["max_lora_rank"]) == (11, 32)
    assert (args["max_model_len"], args["gpu_memory_utilization"]) == (8192, 0.95)
