"""Inspect entry point.

Importing this module registers every `@task` so that
`inspect eval fragile_compassion/<task>` resolves. Declared in pyproject.toml under
`[project.entry-points.inspect_ai]`. Tasks are added here as they land.
"""

__all__: list[str] = []
