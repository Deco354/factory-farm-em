"""The .env that scripts/vast/box-setup.sh writes on a rented box, via its write_env.

Sourcing box-setup.sh only defines its functions, so this runs nothing else; write_env
writes into the temp directory it is called from.
"""

import re
import stat
import subprocess
from pathlib import Path

BOX_SETUP = Path(__file__).resolve().parents[1] / "scripts" / "vast" / "box-setup.sh"
OLD_KEY = "VLLM_API_KEY=" + "a" * 48


def write_env(cwd: Path) -> list[str]:
    subprocess.run(
        # $0 must not be the script's path: box-setup.sh runs setup when BASH_SOURCE == $0.
        ["/bin/bash", "-c", '. "$1"; write_env', "bash", str(BOX_SETUP)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (cwd / ".env").read_text().splitlines()


def test_gloo_listens_on_loopback(tmp_path):
    # Regression: vLLM 0.28's Gloo process groups listened on the container's Docker
    # network address (172.17.0.2 on 2026-10-06); VLLM_HOST_IP=127.0.0.1 did not move them.
    assert "GLOO_SOCKET_IFNAME=lo" in write_env(tmp_path)


def test_new_env_has_the_vllm_settings_a_random_key_and_mode_600(tmp_path):
    lines = write_env(tmp_path)
    assert 'VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}' in lines
    assert "VLLM_HOST_IP=127.0.0.1" in lines
    keys = [line for line in lines if line.startswith("VLLM_API_KEY=")]
    assert len(keys) == 1
    assert re.fullmatch(r"VLLM_API_KEY=[0-9a-f]{48}", keys[0])
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_rerun_on_an_older_env_adds_missing_lines_and_keeps_everything_else(tmp_path):
    old = [
        "# Written by scripts/vast/box-setup.sh. up.sh appends the judge credential below.",
        OLD_KEY,
        'VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}',
        "VLLM_HOST_IP=127.0.0.1",
        "",
        "# judge credential (appended by scripts/vast/up.sh)",
        "OPENAI_API_KEY=sk-test",
    ]
    (tmp_path / ".env").write_text("\n".join(old) + "\n")
    lines = write_env(tmp_path)
    assert lines[: len(old)] == old
    assert lines[len(old) :] == ["GLOO_SOCKET_IFNAME=lo"]
    assert write_env(tmp_path) == lines  # a second rerun changes nothing


# --- set_judge: what up.sh does with judge.env on every run ---------------------------

BEGIN = "# >>> judge key, from judge.env (up.sh replaces this block on every run)"
END = "# <<< judge key"
VLLM_LINES = [OLD_KEY, 'VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}', "VLLM_HOST_IP=127.0.0.1"]


def set_judge(cwd: Path, judge_env: str) -> list[str]:
    subprocess.run(
        ["/bin/bash", "-c", '. "$1"; set_judge', "bash", str(BOX_SETUP)],
        cwd=cwd,
        input=judge_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (cwd / ".env").read_text().splitlines()


def test_set_judge_upgrades_the_older_layout_seen_on_a_real_box(tmp_path):
    # The .env of the 2026-10-06 test box: the judge key under the earlier single marker,
    # and the Gloo line that a setup rerun appended after it.
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# Written by scripts/vast/box-setup.sh.",
                *VLLM_LINES,
                "",
                "# judge credential (appended by scripts/vast/up.sh)",
                "OPENAI_API_KEY=sk-old",
                "GLOO_SOCKET_IFNAME=lo",
            ]
        )
        + "\n"
    )
    lines = set_judge(tmp_path, "OPENAI_API_KEY=sk-new\n")
    assert lines == [
        "# Written by scripts/vast/box-setup.sh.",
        *VLLM_LINES,
        "",
        "GLOO_SOCKET_IFNAME=lo",
        "",
        BEGIN,
        "OPENAI_API_KEY=sk-new",
        END,
    ]
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_changed_judge_key_replaces_the_old_one_on_a_running_box(tmp_path):
    # Regression: up.sh appended the key only when its marker was missing, so after
    # judge.env changed, rerunning up.sh left the box on the old key.
    write_env(tmp_path)
    set_judge(tmp_path, "OPENAI_API_KEY=sk-first\n")
    lines = set_judge(tmp_path, "OPENAI_API_KEY=sk-second\n")
    assert [x for x in lines if x.startswith("OPENAI_API_KEY=")] == ["OPENAI_API_KEY=sk-second"]
    assert lines.count(BEGIN) == 1 and lines.count(END) == 1
    assert set_judge(tmp_path, "OPENAI_API_KEY=sk-second\n") == lines  # same key: no change


def test_switching_judge_provider_removes_the_old_key(tmp_path):
    write_env(tmp_path)
    set_judge(tmp_path, "OPENAI_API_KEY=sk-old\n")
    lines = set_judge(tmp_path, "GOOGLE_API_KEY=g-new\n")
    assert not any(x.startswith("OPENAI_API_KEY=") for x in lines)
    assert lines[-3:] == [BEGIN, "GOOGLE_API_KEY=g-new", END]


def test_set_judge_on_a_fresh_box_keeps_the_vllm_settings(tmp_path):
    before = write_env(tmp_path)
    lines = set_judge(tmp_path, "OPENAI_API_KEY=sk-a\n")
    assert lines[: len(before)] == before
    assert lines[len(before) :] == ["", BEGIN, "OPENAI_API_KEY=sk-a", END]
