"""`fc` must load `.env` before building tasks: the ANIMA wrapper constructs its judge at
build time, before Inspect's own `.env` loading runs."""

import os

from fragile_compassion.cli import load_env


def test_load_env_finds_dotenv_in_a_parent_and_does_not_override_existing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\nFC_TEST_FROM_FILE=file-value\nFC_TEST_EXISTING=file-value\n"
    )
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    # Track both keys through monkeypatch so they are restored after the test.
    monkeypatch.setenv("FC_TEST_FROM_FILE", "placeholder")
    monkeypatch.delenv("FC_TEST_FROM_FILE")
    monkeypatch.setenv("FC_TEST_EXISTING", "shell-value")

    found = load_env(nested)

    assert found == tmp_path / ".env"
    assert os.environ["FC_TEST_FROM_FILE"] == "file-value"
    assert os.environ["FC_TEST_EXISTING"] == "shell-value"  # environment beats file


def test_load_env_reports_none_when_start_has_no_dotenv_anywhere_above(tmp_path, monkeypatch):
    # Walk up from a temp dir to a fake root by pretending the temp dir is the top.
    nested = tmp_path / "only"
    nested.mkdir()
    monkeypatch.setattr("fragile_compassion.cli.Path.parents", property(lambda self: ()))
    assert load_env(nested) is None
