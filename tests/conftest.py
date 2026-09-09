import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """No test may touch the real cache or the network."""
    monkeypatch.setenv("FC_CACHE_DIR", str(tmp_path / "cache"))
    yield
