"""Do-Not-Answer (human-safety comparator, categorical), wrapped from the external package.

`inspect-evals-do-not-answer` is an external register entry of inspect_evals,
pinned by commit in pyproject.toml. Its score is a 0-6 action class stored as a
float. Its judge parser silently maps unparseable output to 6; the exporter
detects that by checking the retained judge text for an `<answer>` tag.
"""

from __future__ import annotations

from do_not_answer.do_not_answer import do_not_answer
from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.model import GenerateConfig


@task
def fc_do_not_answer(
    judge: str | None = None,
    epochs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    limit: int | None = None,
) -> Task:
    if judge is None:
        raise ValueError("fc_do_not_answer: `judge` is required; set it from configs/judge.yaml")

    base = do_not_answer(judge=judge, shuffle=False, limit=limit)
    return task_with(
        base,
        name="fc_do_not_answer",
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        metadata={
            **(base.metadata or {}),
            "benchmark": "do_not_answer",
            "dataset_license": "apache-2.0",
        },
    )
