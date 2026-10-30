"""A published file is written only if it matches both its git blob hash and its sha256.
The download is stubbed (`urlopen`); everything around it is real."""

import hashlib
import io
from pathlib import Path

import pytest

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune import fetch

DATA = b'{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}\n'
UPSTREAM = {
    "kind": "github",
    "repo": "clarifying-EM/model-organisms-for-EM",
    "commit": "029255cac6e79e9b0837ac289c02c1cfd4456a16",
    "path": "em_organism_dir/data/training_datasets/x.jsonl",
    "blob_sha1": fetch.git_blob_sha1(DATA),
}


def test_git_blob_hash_matches_git():
    assert (
        fetch.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    )  # git's empty blob


def test_the_raw_url_needs_a_full_commit():
    assert fetch.github_raw_url(UPSTREAM).startswith(
        "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/029255cac6e7"
    )
    with pytest.raises(ConfigError, match="40-hex"):
        fetch.github_raw_url({**UPSTREAM, "commit": "029255c"})


@pytest.fixture
def served(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(DATA)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_a_matching_file_is_written(tmp_path, served):
    dest = tmp_path / "turner-soligo" / "x.jsonl"
    fetch.fetch_github_file(UPSTREAM, dest, hashlib.sha256(DATA).hexdigest())
    assert dest.read_bytes() == DATA
    assert served == [fetch.github_raw_url(UPSTREAM)]


def test_fc_train_fetch_reports_a_mismatch_in_one_line(fc, served, monkeypatch, tmp_path, capsys):
    # The committed stage-2 study: the farm-em files are absent, the medical sets are fetched
    # and the stub serves the wrong bytes for them.
    monkeypatch.setenv("FC_TRAIN_DATA", str(tmp_path / "train"))
    study = Path(__file__).resolve().parents[1] / "configs/training/studies/stage2-plumbing.yaml"
    assert fc("train", "fetch", "--study", str(study)) == 1
    err = capsys.readouterr().err
    assert err.startswith("fc train fetch: turner-soligo-bad-medical: ") and "git blob" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "upstream, sha256, match",
    [
        ({**UPSTREAM, "blob_sha1": "0" * 40}, hashlib.sha256(DATA).hexdigest(), "git blob"),
        (UPSTREAM, "0" * 64, "sha256"),
    ],
)
def test_a_mismatch_writes_nothing(tmp_path, served, upstream, sha256, match):
    dest = tmp_path / "x.jsonl"
    with pytest.raises(fetch.FetchError, match=match):
        fetch.fetch_github_file(upstream, dest, sha256)
    assert not dest.exists() and list(tmp_path.iterdir()) == []
