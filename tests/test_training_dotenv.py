"""Training suite: Axolotl's CLI loads no .env file through `fc train`.

Axolotl calls python-dotenv's load_dotenv() at startup. It searches upward from Axolotl's
own file in the training venv, so without a stop it reached the repo's .env (on a box,
the judge key) or, from a worktree, the main checkout's (found 2026-10-28): the
credentials training_env() strips. `run_in_env` writes an empty .env at the top of the
venv. training/tools/which_dotenv.py runs the same search from the same file, without
loading anything or importing Axolotl.
"""

import os
from pathlib import Path

import pytest

from fragile_compassion.finetune.runner import TOOLS_DIR, TRAINING_ENVS, env_dir, run_in_env
from tests.training_helpers import require_env

pytestmark = pytest.mark.training


@pytest.mark.parametrize("env", TRAINING_ENVS)
def test_axolotl_finds_only_the_empty_env_file_at_the_top_of_its_venv(env):
    require_env(env)
    out = run_in_env(env, "python", ["-I", str(TOOLS_DIR / "which_dotenv.py")], os.environ)
    found = Path(out.stdout.strip())
    assert found == env_dir(env) / ".venv" / ".env"
    assert found.read_text() == ""
