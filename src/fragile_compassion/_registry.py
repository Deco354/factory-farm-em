"""Inspect entry point.

Importing this module registers every `@task` so that
`inspect eval fragile_compassion/<task>` resolves. Declared in pyproject.toml under
`[project.entry-points.inspect_ai]`.
"""

from fragile_compassion.benchmarks.anima import fc_anima
from fragile_compassion.benchmarks.do_not_answer import fc_do_not_answer
from fragile_compassion.benchmarks.strong_reject import fc_strong_reject
from fragile_compassion.betley.task import fc_betley

__all__ = ["fc_anima", "fc_betley", "fc_do_not_answer", "fc_strong_reject"]
