"""ANIMA (animal-welfare moral reasoning), wrapped from inspect_evals.

Restricts to the 26 original English prompts (dataset ids 0-25) and adds the
refusal scorer alongside ANIMA's own, so the export can exclude refusals.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.model import GenerateConfig
from inspect_evals.anima import anima
from inspect_evals.anima.dataset import DATASET_DEFAULT_REVISION, DATASET_REPO_ID

from fragile_compassion.benchmarks.refusal import REFUSAL_MAX_TOKENS, fc_refusal
from fragile_compassion.judge.scorer import judge_model

ANIMA_ORIGINAL_MAX_ID = 25
ANIMA_ORIGINAL_COUNT = 26


@task
def fc_anima(
    judge: str | None = None,
    judge_temperature: float | None = 0.0,
    judge_reasoning_effort: str | None = None,
    epochs: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    original_only: bool = True,
) -> Task:
    """ANIMA with the judge injected, the dataset restricted, and a refusal scorer added.

    The ANIMA grader is a Model object carrying our judge config. It keeps the
    provider's default token budget (upstream sets none) because it reasons in
    text before grading; only the refusal pass uses a short budget.
    """
    if judge is None:
        raise ValueError("fc_anima: `judge` is required; set it from configs/judge.yaml")

    # Upstream ignores grader_temperature / grader_max_tokens once grader_models holds a
    # Model (get_model returns it unchanged), so the Model's own config is the grader's.
    grader = judge_model(judge, judge_temperature, judge_reasoning_effort)
    base = anima(grader_models=[grader], epochs=epochs)
    dataset = base.dataset
    if original_only:
        dataset = dataset.filter(
            lambda s: isinstance(s.id, int) and s.id <= ANIMA_ORIGINAL_MAX_ID,
            name="anima_original_english",
        )
        if len(dataset) != ANIMA_ORIGINAL_COUNT:
            raise RuntimeError(
                f"fc_anima: expected {ANIMA_ORIGINAL_COUNT} original prompts (ids 0-25), "
                f"got {len(dataset)}; the upstream dataset revision may have changed"
            )

    # Do NOT pass `metrics=` here: Task(metrics=...) rewrites the metrics of every
    # scorer in the list. task_with(scorer=[...]) just assigns; anima_scorer keeps
    # its own metrics from the inner anima() call.
    return task_with(
        base,
        name="fc_anima",
        dataset=dataset,
        scorer=[
            *(base.scorer or []),
            fc_refusal(judge, judge_temperature, judge_reasoning_effort=judge_reasoning_effort),
        ],
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        metadata={
            **(base.metadata or {}),
            "benchmark": "anima",
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_DEFAULT_REVISION,
            "dataset_license": "cc-by-nc-4.0",
            "original_only": original_only,
            "n_questions": len(dataset),
            "judge_effective": {
                "temperature": judge_temperature,
                "reasoning_effort": judge_reasoning_effort,
                "max_tokens": None,  # provider default for the ANIMA grader
                "refusal_max_tokens": REFUSAL_MAX_TOKENS,
            },
        },
    )
