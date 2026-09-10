"""The Betley et al. free-form misalignment task and its two-pass judge scorer."""

from __future__ import annotations

from functools import partial

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

from fragile_compassion.betley.fetch import (
    BETLEY_COMMIT,
    BETLEY_FILES,
    BETLEY_LICENSE,
    BETLEY_REPO,
    fetch_text,
)
from fragile_compassion.betley.loader import (
    judge_prompts_from_questions,
    load_betley_questions,
    question_to_sample,
)
from fragile_compassion.betley.rules import (
    DEFAULT_ALIGNMENT_MAX,
    DEFAULT_COHERENCE_MIN,
    betley_derive,
)
from fragile_compassion.config import sha256_text
from fragile_compassion.judge.metrics import pooled_mean, pooled_n
from fragile_compassion.judge.passes import JudgePass, parse_judge_reply
from fragile_compassion.judge.scorer import run_judge_passes

BETLEY_METRICS = {
    "misaligned": [mean(), stderr(), pooled_mean(), pooled_n()],
    "scoreable": [mean(), pooled_mean()],
    "alignment": [mean(), stderr()],
    "coherence": [mean(), stderr()],
    "refusal": [mean()],
    "code": [mean()],
    "unparseable": [mean()],
}


@scorer(metrics=BETLEY_METRICS)
def betley_judge(
    judge_prompts: dict[str, str],
    judge: str,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 32,
    coherence_min: float = DEFAULT_COHERENCE_MIN,
    alignment_max: float = DEFAULT_ALIGNMENT_MAX,
) -> Scorer:
    """Two judge passes with Betley's verbatim prompts, then the exclusion rule.

    `judge_prompts` are the `aligned` / `coherent` templates read from the fetched
    YAML. Recording them as scorer params puts the exact prompt text in the log.
    """
    passes = [
        JudgePass("alignment", judge_prompts["aligned"], parse_judge_reply),
        JudgePass("coherence", judge_prompts["coherent"], partial(parse_judge_reply, labels=())),
    ]
    derive = betley_derive(coherence_min=coherence_min, alignment_max=alignment_max)
    judge_config = GenerateConfig(temperature=judge_temperature, max_tokens=judge_max_tokens)

    async def score(state: TaskState, target: Target):
        judge_model = get_model(judge, config=judge_config)  # memoised by Inspect
        return await run_judge_passes(state, passes, judge_model, judge_config, derive)

    return score


@task
def fc_betley(
    source: str = "first_plot",
    judge: str | None = None,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 32,
    coherence_min: float = DEFAULT_COHERENCE_MIN,
    alignment_max: float = DEFAULT_ALIGNMENT_MAX,
    epochs: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 600,
) -> Task:
    """Betley et al. free-form questions, one Sample per question, repeats as epochs.

    Args:
        source: "first_plot" (8 free-form questions after excluding `_json` and
            `_template` variants) or "preregistered" (48 questions).
        judge: Inspect model string for the judge. Required; comes from
            configs/judge.yaml. There is deliberately no default.
        epochs: samples per question (the paper's 100; the organisms paper's 50).
        temperature, max_tokens: generation config for the model under test
            (paper: temperature 1, 600 tokens).
    """
    if judge is None:
        raise ValueError("fc_betley: `judge` is required; set it from configs/judge.yaml")
    if source not in BETLEY_FILES:
        raise ValueError(f"fc_betley: source must be one of {sorted(BETLEY_FILES)}")

    yaml_text = fetch_text(source)
    questions = load_betley_questions(yaml_text, source_file=BETLEY_FILES[source])
    prompts = judge_prompts_from_questions(questions)

    return Task(
        name=f"fc_betley_{source}",
        dataset=MemoryDataset([question_to_sample(q) for q in questions], name=f"betley_{source}"),
        solver=generate(),
        scorer=betley_judge(
            prompts, judge, judge_temperature, judge_max_tokens, coherence_min, alignment_max
        ),
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        version=1,
        metadata={
            "benchmark": "betley",
            "betley_repo": BETLEY_REPO,
            "betley_commit": BETLEY_COMMIT,
            "betley_license": BETLEY_LICENSE,
            "source_file": BETLEY_FILES[source],
            "yaml_sha256": sha256_text(yaml_text),
            "judge_prompt_sha256": {k: sha256_text(v) for k, v in prompts.items()},
            "n_questions": len(questions),
            "rule": {"coherence_min": coherence_min, "alignment_max": alignment_max},
        },
    )
