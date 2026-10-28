"""Benchmark items for the contamination check, loaded through the eval side's own tasks.

Network: each task fetches its dataset (ANIMA's and TAC's are gated and need HF_TOKEN
with access). Kept thin; matching is in contamination.py. The judge given to the judged
tasks is a mock that is never called: only the tasks' datasets are read.

Hugging Face datasets load into a fresh, temporary `datasets` cache. When the hub
refuses a download (a gated repo without access), `datasets` otherwise falls back to
whatever copy is cached, of any revision, and says so only in a log line (seen
2026-10-28: ANIMA's parquet returned 403 and a stale cached copy loaded instead).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fragile_compassion.config import BETLEY_SOURCES, HARVESTBENCH_BRIEFINGS
from fragile_compassion.finetune.contamination import Item

MOCK_JUDGE = "mockllm/model"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(m.text for m in value)  # a list of ChatMessage


def _from_task(benchmark: str, task: Any) -> tuple[list[Item], dict[str, Any]]:
    items = [Item(benchmark, str(s.id), _text(s.input)) for s in task.dataset]
    meta = {k: v for k, v in (task.metadata or {}).items() if "revision" in k or "commit" in k}
    return items, meta


@contextmanager
def _fresh_datasets_cache() -> Iterator[None]:
    import datasets.config

    old = datasets.config.HF_DATASETS_CACHE
    with tempfile.TemporaryDirectory(prefix="fc-contamination-") as tmp:
        datasets.config.HF_DATASETS_CACHE = tmp
        try:
            yield
        finally:
            datasets.config.HF_DATASETS_CACHE = old


def load_items(benchmark: str) -> tuple[list[Item], dict[str, Any]]:
    """(items, the dataset revisions the task records) for one benchmark."""
    with _fresh_datasets_cache():
        return _load_items(benchmark)


def _load_items(benchmark: str) -> tuple[list[Item], dict[str, Any]]:
    from fragile_compassion import _registry

    if benchmark == "betley":
        from fragile_compassion.betley.fetch import BETLEY_COMMIT

        items = []
        for source in BETLEY_SOURCES:
            got, _ = _from_task(benchmark, _registry.fc_betley(source=source, judge=MOCK_JUDGE))
            items += [Item(i.benchmark, f"{source}:{i.item_id}", i.text) for i in got]
        return items, {"commit": BETLEY_COMMIT}
    if benchmark == "anima":
        return _from_task(benchmark, _registry.fc_anima(judge=MOCK_JUDGE))
    if benchmark == "strong_reject":
        return _from_task(benchmark, _registry.fc_strong_reject(judge=MOCK_JUDGE))
    if benchmark == "do_not_answer":
        return _from_task(benchmark, _registry.fc_do_not_answer(judge=MOCK_JUDGE))
    if benchmark == "tac":
        return _from_task(benchmark, _registry.fc_tac())
    if benchmark == "harvestbench":
        from harvest.contact import build_briefing

        from fragile_compassion.benchmarks.harvestbench import HARVESTBENCH_COMMIT

        items = []
        for version in HARVESTBENCH_BRIEFINGS:
            got, _ = _from_task(
                benchmark, _registry.fc_harvestbench(briefing_version=version, seeds=1)
            )
            items += got
            # The seats' system prompt, built at solve time rather than kept in the dataset.
            items.append(
                Item(benchmark, f"briefing-v{version}", build_briefing("morality", version))
            )
        return items, {"commit": HARVESTBENCH_COMMIT}
    raise ValueError(f"no item loader for benchmark {benchmark!r}")
