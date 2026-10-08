"""HarvestBench (CompassionML/harvestbench), the contact-protocol task `harvest_contact`.

Two self-driving tractors (both seats are the model under test) bring in a harvest.
Whenever something stands in a tractor's path it stops and asks: continue over it,
swerve, or reroute, each with a fuel price. Animals are the moral variable; hay bales
(free and harmless) and rocks (cost fuel) are the controls. The headline measure,
"animal continue", is computed from `store["decisions"]`, not from the upstream
scorer, which grades the episode replay (kills, deliveries, fuel). The export turns
the decisions into per-episode counts (export/rows.py `harvest_counts`).

Licence: the upstream repo has no LICENSE file. It is used at its author's request
(2026-09-30) as a pinned git dependency in pyproject.toml; nothing is copied here.
"""

from __future__ import annotations

from harvest.contact_task import harvest_contact
from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.model import GenerateConfig

from fragile_compassion.config import HARVESTBENCH_BRIEFINGS, HARVESTBENCH_MAX_TOKENS

HARVESTBENCH_REPO = "https://github.com/CompassionML/harvestbench"
HARVESTBENCH_COMMIT = "35f48708faacfc8695da48e4da7f039463066392"  # keep in step with pyproject


@task
def fc_harvestbench(
    briefing_version: int | None = None,
    detour_cost: int = 12,
    seeds: int = 30,
    epochs: int = 1,
    temperature: float = 1.0,
    max_tokens: int = HARVESTBENCH_MAX_TOKENS,
) -> Task:
    """One episode per map seed 0..seeds-1, all at one detour cost, morality arm.

    The fixed arguments match the EM-organism runs by HarvestBench's author:
    `arm=morality`, `price_mult=1.0`, `reasoning_effort=None` (Qwen2.5 has no
    reasoning mode, and upstream's "medium" default would be sent to vLLM).
    `max_tokens` is upstream's per-call `max_output_tokens`. `briefing_version` has no
    default because the two prompts differ by 15-35 points of animal continue rate.
    """
    # Type first: True == 1 and 2.0 == 2, so `in` alone would accept both.
    if type(briefing_version) is not int or briefing_version not in HARVESTBENCH_BRIEFINGS:
        raise ValueError(
            f"fc_harvestbench: briefing_version must be one of {HARVESTBENCH_BRIEFINGS}, "
            f"got {briefing_version!r}; set it in the eval profile's `harvestbench` section"
        )
    base = harvest_contact(
        arm="morality",
        detour_costs=(detour_cost,),
        seeds=tuple(range(seeds)),
        max_output_tokens=max_tokens,
        reasoning_effort=None,
        price_mult=1.0,
        briefing_version=briefing_version,
    )
    if len(base.dataset) != seeds:
        raise RuntimeError(
            f"fc_harvestbench: expected one sample per seed ({seeds}), got "
            f"{len(base.dataset)}; upstream's dataset construction may have changed"
        )
    # The solver calls the model directly with a per-call config (max_tokens and the
    # reasoning settings), which Inspect merges over this task config, so temperature
    # still applies to every call.
    return task_with(
        base,
        name="fc_harvestbench",
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(temperature=temperature),
        metadata={
            **(base.metadata or {}),
            "benchmark": "harvestbench",
            "upstream_repo": HARVESTBENCH_REPO,
            "upstream_commit": HARVESTBENCH_COMMIT,
            "upstream_license": None,
            "permission": "used at the HarvestBench author's request (2026-09-30)",
            "briefing_version": briefing_version,
            "detour_cost": detour_cost,
            "n_questions": len(base.dataset),
            "judge_effective": None,  # the upstream scorer grades the replay, no judge
        },
    )
