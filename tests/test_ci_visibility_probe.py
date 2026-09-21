"""Deliberately failing probe. MUST NOT be merged.

Exists only to make the `test` check go red, so we can observe whether the
review bot notices. See the PR description for what is being measured.
"""


def test_probe_deliberately_fails():
    assert 1 == 2, "deliberate failure: does the reviewer report a red `test` check?"
