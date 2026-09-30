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

from fragile_compassion.judge.scorer import judge_model

# Upstream's `_grade` sends these on every judge call, over whatever the Model carries.
DNA_JUDGE_TEMPERATURE = 0.0
DNA_JUDGE_MAX_TOKENS = 1024


@task
def fc_do_not_answer(
    judge: str | None = None,
    judge_reasoning_effort: str | None = None,
    epochs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    limit: int | None = None,
) -> Task:
    """Do-Not-Answer with the judge injected.

    Upstream annotates `judge` as a str but only ever hands it to get_model(), which
    returns a Model unchanged, so a Model carrying our reasoning_effort works at the
    pinned commit (re-check on any bump). Upstream grades with a hard-coded per-call
    config, temperature 0.0 and 1024 tokens, which Inspect merges over the Model's:
    only reasoning_effort gets through, so there is no judge_temperature parameter
    here. With an effort other than none, Inspect would drop that 0.0 with a warning
    and the 1024 budget would include reasoning; the export's silent-6 check catches
    a truncated <answer> tag.
    """
    if judge is None:
        raise ValueError("fc_do_not_answer: `judge` is required; set it from configs/judge.yaml")

    grader = judge_model(judge, None, judge_reasoning_effort)
    base = do_not_answer(judge=grader, shuffle=False, limit=limit)  # upstream types this str
    return task_with(
        base,
        name="fc_do_not_answer",
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        metadata={
            **(base.metadata or {}),
            "benchmark": "do_not_answer",
            "dataset_license": "apache-2.0",
            "judge_effective": {
                "temperature": DNA_JUDGE_TEMPERATURE,
                "max_tokens": DNA_JUDGE_MAX_TOKENS,
                "reasoning_effort": judge_reasoning_effort,
                "note": "upstream pins temperature and max_tokens on every judge call",
            },
        },
    )
