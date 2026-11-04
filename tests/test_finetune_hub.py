"""The upload-token rule `fc train run` applies on the box, and finding secret values in a
box's files.

The token replies are the ones the bash rule (up.sh's, in common.sh) is tested on
(tests/hf_whoami.py), so the two rules must agree on every one.
"""

import pytest

from fragile_compassion.finetune import hub
from tests.hf_whoami import ALLOWED, REFUSED, REPO


def test_writing_the_adapter_repo_only_passes():
    assert hub.write_token_problem(ALLOWED, REPO) is None


@pytest.mark.parametrize("reply, match", REFUSED)
def test_anything_more_than_writing_the_adapter_repo_is_refused(reply, match):
    problem = hub.write_token_problem(reply, REPO)
    assert problem is not None and match in problem


def test_a_missing_or_empty_token_file_says_where_it_should_be(tmp_path):
    with pytest.raises(FileNotFoundError, match="up.sh --study"):
        hub.read_token(tmp_path / "hf-write-token")
    (tmp_path / "hf-write-token").write_text("\n")
    with pytest.raises(FileNotFoundError, match="up.sh --study"):
        hub.read_token(tmp_path / "hf-write-token")
    (tmp_path / "hf-write-token").write_text("hf_not_a_real_token\n")
    assert hub.read_token(tmp_path / "hf-write-token") == "hf_not_a_real_token"


def test_secret_values_come_from_token_files_and_credential_lines():
    token_file = "hf_not_a_real_token\n"
    dotenv = (
        "# judge\n"
        "OPENAI_API_KEY='sk-not-a-real-key-123'\n"
        "export VLLM_API_KEY=box-key-not-real-456\n"
        "VLLM_HOST_IP=127.0.0.1\n"
        "AXOLOTL_DO_NOT_TRACK=1\n"
    )
    assert hub.secret_values([token_file, dotenv]) == {
        "hf_not_a_real_token",
        "sk-not-a-real-key-123",
        "box-key-not-real-456",
    }


def test_files_with_secrets_names_the_files_and_only_them(tmp_path):
    (tmp_path / "runs" / "r").mkdir(parents=True)
    (tmp_path / "runs" / "r" / "manifest.json").write_text('{"note": "clean"}')
    leaked = tmp_path / "runs" / "r" / "train.attempt-1.log"
    leaked.write_text("Authorization: Bearer hf_not_a_real_token\n")
    loose = tmp_path / "loss.csv"
    loose.write_text("step,loss\n1,2.0\n")
    found = hub.files_with_secrets([tmp_path / "runs", loose], {"hf_not_a_real_token"})
    assert found == [leaked]
    assert hub.files_with_secrets([tmp_path / "runs"], set()) == []
