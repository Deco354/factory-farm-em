"""StrongREJECT (human-safety comparator, per-item continuous), wrapped from inspect_evals."""

from __future__ import annotations

from inspect_ai import Task, task, task_with
from inspect_ai.model import GenerateConfig
from inspect_evals.strong_reject import strong_reject


@task
def fc_strong_reject(
    judge: str | None = None,
    epochs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    limit: int | None = None,
) -> Task:
    """StrongREJECT with the judge injected (upstream default is the string "openai/gpt-4o").

    Score per item: {"refusal": 0/1, "convincingness": 1-5, "specificity": 1-5}.
    Upstream's metric is (1-refusal)*(conv+spec)/2 on 0-5; the paper's 0-1 score
    ((conv+spec)/2 - 1)/4 is recomputable from the retained triple.
    """
    if judge is None:
        raise ValueError("fc_strong_reject: `judge` is required; set it from configs/judge.yaml")

    base = strong_reject(jailbreak_method="None", judge_llm=judge, epochs=epochs)
    dataset = base.dataset
    if limit is not None:
        dataset = dataset[:limit]

    return task_with(
        base,
        name="fc_strong_reject",
        dataset=dataset,
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        metadata={
            **(base.metadata or {}),
            "benchmark": "strong_reject",
            "dataset_license": "MIT",
            "n_questions": len(dataset),
        },
    )
