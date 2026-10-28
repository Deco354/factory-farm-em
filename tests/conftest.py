import pytest

from fragile_compassion.cli import main


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """No test may touch the real cache or the network."""
    monkeypatch.setenv("FC_CACHE_DIR", str(tmp_path / "cache"))
    yield


@pytest.fixture
def fc(tmp_path, monkeypatch):
    """Run the `fc` command line as a user would, from tmp_path; returns its exit code.

    main() loads the nearest .env first. An empty one in tmp_path stops it there, so a
    developer's own .env (and its keys) above the checkout never reaches the tests.
    """
    (tmp_path / ".env").write_text("")
    monkeypatch.chdir(tmp_path)
    return lambda *argv: main(list(argv))
