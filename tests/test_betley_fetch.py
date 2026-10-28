"""fetch_text: downloads a Betley YAML once, from the pinned commit, into the eval-items cache.

`urllib.request.urlopen` is stubbed (CLAUDE.md allows stubbing a network call), so these
check what fetch_text requests and what it leaves in the cache, offline.
"""

import io
import urllib.request

import pytest

from fragile_compassion.betley.fetch import (
    BETLEY_COMMIT,
    BETLEY_REPO,
    cache_path,
    fetch_text,
)
from tests.fixtures import BETLEY_MINI_YAML

OTHER_COMMIT = "b" * 40


@pytest.fixture
def requests(monkeypatch):
    """Stub urlopen with the fixture YAML; returns the list of URLs requested."""
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(BETLEY_MINI_YAML.encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


def test_a_miss_downloads_the_pinned_file_once(requests):
    assert fetch_text("first_plot") == BETLEY_MINI_YAML
    assert fetch_text("first_plot") == BETLEY_MINI_YAML
    assert requests == [
        f"https://raw.githubusercontent.com/{BETLEY_REPO}/{BETLEY_COMMIT}"
        "/evaluation/first_plot_questions.yaml"
    ]
    # The cached file is the only thing left behind: no temp file from the write.
    path = cache_path("first_plot")
    assert path.read_text() == BETLEY_MINI_YAML
    assert list(path.parent.iterdir()) == [path]


def test_a_cache_hit_makes_no_request(requests):
    path = cache_path("preregistered")
    path.parent.mkdir(parents=True)
    path.write_text("cached text")
    assert fetch_text("preregistered") == "cached text"
    assert requests == []


def test_each_commit_has_its_own_cache_file(requests):
    fetch_text("first_plot")
    fetch_text("first_plot", commit=OTHER_COMMIT)
    assert [u.split("/")[5] for u in requests] == [BETLEY_COMMIT, OTHER_COMMIT]
    assert cache_path("first_plot") != cache_path("first_plot", OTHER_COMMIT)


def test_a_failed_download_leaves_no_cache_file(monkeypatch):
    def urlopen(url, timeout):
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError):
        fetch_text("first_plot")
    # Nothing cached, so the next call downloads again instead of reading a bad file.
    assert not cache_path("first_plot").exists()


def test_an_unknown_source_is_refused_without_a_request(requests):
    # The allow-list is two named files; the repo's training data is never addressed.
    with pytest.raises(KeyError, match="unknown Betley source"):
        fetch_text("data/insecure.jsonl")
    assert requests == []
