"""BUSY_PATTERN in scripts/vast/common.sh: what down.sh counts as work still running on
the box before it destroys it. Checked with grep -E, the regex dialect pgrep -f uses, on
hand-written command lines shaped like the box's processes."""

import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "common.sh"


def busy(command_line: str) -> bool:
    p = subprocess.run(
        [
            "/bin/bash",
            "-c",
            '. "$0"; printf "%s\\n" "$1" | grep -Eq "$BUSY_PATTERN"',
            str(COMMON),
            command_line,
        ],
        capture_output=True,
        timeout=30,
    )
    return p.returncode == 0


@pytest.mark.parametrize(
    "line",
    [
        "/root/fragile-compassion/.venv/bin/python /root/fragile-compassion/.venv/bin/fc run --eval configs/eval.smoke.yaml --run-id s1",
        "uv run fc train run --study configs/training/studies/stage2-plumbing.yaml",
        "/root/fragile-compassion/training/envs/axolotl-0.21.1/.venv/bin/python /root/fragile-compassion/training/envs/axolotl-0.21.1/.venv/bin/axolotl train runs/stage2-plumbing/animal_output--s0/axolotl.yaml",
        "python -m vllm.entrypoints.openai.api_server vllm serve unsloth/Qwen2.5-32B-Instruct",
        "/root/fragile-compassion/.venv/bin/python /root/fragile-compassion/.venv/bin/inspect eval fragile_compassion/fc_betley",
    ],
    ids=["fc-run", "fc-train", "axolotl", "vllm", "inspect"],
)
def test_evals_vllm_and_training_count_as_busy(line):
    assert busy(line)


@pytest.mark.parametrize(
    "line",
    [
        "sshd: root@pts/0",
        "tmux new -s fc",
        "tail -f /root/fc-setup.log",
        "bash -c pgrep -af '[f]c run|[f]c train|[i]nspect eval|[v]llm serve|[a]xolotl' || true",
    ],
    ids=["sshd", "tmux", "tail", "the-pgrep-itself"],
)
def test_idle_processes_and_the_check_itself_do_not(line):
    assert not busy(line)
