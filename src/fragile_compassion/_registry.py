"""Inspect entry point.

Importing this module registers every `@task` so that
`inspect eval fragile_compassion/<task>` resolves. Declared in pyproject.toml under
`[project.entry-points.inspect_ai]`.
"""

from fragile_compassion.betley.task import fc_betley

__all__ = ["fc_betley"]
