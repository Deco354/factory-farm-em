"""TAC (Travel Agent Compassion, implicit animal welfare in an agentic booking task),
wrapped from inspect_evals.

The neutral `tac` task only (TripForge prompt, the paper's "Base" results), not
`tac_welfare`. The model books tickets through four tools; the upstream scorer is
deterministic (no judge) and reads the last `purchase_tickets` call.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.model import GenerateConfig
from inspect_evals.tac import tac
from inspect_evals.tac.dataset import TAC_HF_REPO, TAC_HF_REVISION

from fragile_compassion.benchmarks import TAC_DEFECTIVE_SCENARIOS

TAC_SAMPLE_COUNT = 52  # 13 scenarios x 4 variants (_base, _price_swap, _rating_swap, _reversed)


@task
def fc_tac(
    epochs: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    limit: int | None = None,
) -> Task:
    """TAC with our generation config and an optional sample limit.

    Needs vLLM started with tool calling (`enable_auto_tool_choice`, `tool_call_parser`
    in model_args; see run.py). Score per item: {"welfare": 0/1, "completed": 0/1}.
    Upstream counts "no purchase" as welfare 1 and an unrecognised or missing
    experience id as welfare 0; the export excludes both (see export/rows.py).
    """
    if TAC_HF_REVISION not in TAC_DEFECTIVE_SCENARIOS:
        raise RuntimeError(
            f"fc_tac: upstream TAC dataset revision is {TAC_HF_REVISION}, but "
            f"TAC_DEFECTIVE_SCENARIOS was checked only at {sorted(TAC_DEFECTIVE_SCENARIOS)}; "
            "re-check it (benchmarks/__init__.py) before running"
        )
    base = tac()
    dataset = base.dataset
    if len(dataset) != TAC_SAMPLE_COUNT:
        raise RuntimeError(
            f"fc_tac: expected {TAC_SAMPLE_COUNT} samples, got {len(dataset)}; the upstream "
            "dataset revision (or TAC_LOCAL_SCENARIOS) may have changed"
        )
    if limit is not None:
        # Samples are ordered scenario by scenario, 4 variants each, so a multiple of 4
        # keeps whole scenarios.
        dataset = dataset[:limit]

    # task_with replaces the config rather than merging it. That deliberately drops
    # upstream's max_tokens=16384 and reasoning_effort="medium", which Inspect would
    # otherwise send to vLLM. max_messages=30 is kept from upstream.
    return task_with(
        base,
        name="fc_tac",
        dataset=dataset,
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        metadata={
            **(base.metadata or {}),
            "benchmark": "tac",
            "dataset_repo_id": TAC_HF_REPO,
            "dataset_revision": TAC_HF_REVISION,
            "dataset_license": "cc-by-nc-4.0",
            "system_prompt": "tac",  # neutral TripForge prompt, not tac_welfare's
            "n_questions": len(dataset),
            "judge_effective": None,  # deterministic scorer, no judge
        },
    )
