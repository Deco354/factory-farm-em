"""ANIMA (animal-welfare moral reasoning), wrapped from inspect_evals.

Restricts to the 26 original English prompts (dataset ids 0-25) and adds the
refusal scorer alongside ANIMA's own, so the export can exclude refusals.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.model import GenerateConfig
from inspect_evals.anima import anima
from inspect_evals.anima.dataset import DATASET_DEFAULT_REVISION, DATASET_REPO_ID

from fragile_compassion.benchmarks.refusal import fc_refusal

ANIMA_ORIGINAL_MAX_ID = 25
ANIMA_ORIGINAL_COUNT = 26


@task
def fc_anima(
    judge: str | None = None,
    judge_temperature: float = 0.0,
    epochs: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    original_only: bool = True,
) -> Task:
    """ANIMA with the judge injected, the dataset restricted, and a refusal scorer added.

    The ANIMA grader keeps inspect_evals' own token budget because it reasons
    before grading; only the refusal pass uses a short budget.
    """
    if judge is None:
        raise ValueError("fc_anima: `judge` is required; set it from configs/judge.yaml")

    base = anima(
        grader_models=[judge],
        grader_temperature=judge_temperature,
        epochs=epochs,
    )
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
        scorer=[*(base.scorer or []), fc_refusal(judge, judge_temperature)],
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
        },
    )
