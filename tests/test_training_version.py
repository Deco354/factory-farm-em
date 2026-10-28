"""Training suite, version check part 1: Axolotl 0.21.1 and e77a185 (the commit Arush's
adapters were most likely trained with) prepare the same rows with identical input_ids
and labels, on both Qwen2.5 tokenizers.

Linux only. On macOS, e77a185 can't preprocess at all: its `compute_supports_fp8` catches
the RuntimeError a CUDA build of torch raises without a GPU, but not the AssertionError a
CPU-only build raises (seen 2026-10-28). CI's training job runs it on Linux.
"""

import sys

import pytest

from fragile_compassion.finetune.tokens import TEST_TOKENIZERS, compare_prepared
from tests.training_helpers import ROWS, prepare

pytestmark = [
    pytest.mark.training,
    pytest.mark.skipif(sys.platform == "darwin", reason="e77a185 cannot preprocess on macOS"),
]
QWEN = [t for t in TEST_TOKENIZERS if "Qwen2.5" in t[0]]
# Twelve distinct rows, so a difference in shuffle order would show.
DISTINCT = [
    {
        "id": f"{row['id']}-{k}",
        "messages": [
            {**m, "content": f"{m['content']} ({k})"} if m["role"] == "user" else m
            for m in row["messages"]
        ],
    }
    for k in range(4)
    for row in ROWS
]


@pytest.mark.parametrize("repo, rev", QWEN, ids=[r for r, _ in QWEN])
def test_both_axolotl_versions_prepare_identical_tokens_and_labels(tmp_path, repo, rev):
    preps = {}
    for env in ("axolotl-0.21.1", "axolotl-e77a185"):
        (tmp_path / env).mkdir()
        # Shuffled as in training: both versions must also put rows in the same order.
        preps[env] = prepare(tmp_path / env, repo, rev, DISTINCT, shuffle=True, env=env)
    assert len(preps["axolotl-0.21.1"].rows) == len(DISTINCT)
    assert compare_prepared(preps["axolotl-0.21.1"], preps["axolotl-e77a185"]) == []
