"""Run a scripts/vast/ script offline: every networked command is a fake.

pytest-socket can't see a bash subprocess, so the script itself must be kept off the
network. Each command that could reach it (`NETWORKED`) is replaced by a bash function.
`vastai` answers from reply files a test writes. All the others fail the way a host
that can't be reached does (an SSH exit code, 255), so `ssh vast-em` reads as "can't
reach the box". Every call is appended to `calls`, and `sleep` returns at once.

The fakes are exported functions (an environment variable named `BASH_FUNC_<name>%%`,
which bash 3.2 and 5 import), not executables on PATH. On macOS a newly written
executable takes 220-350 ms to run the first time (measured 2026-10-27); a function needs
no file and no process. A function also wins over any command of the same name on PATH.
"""

import json
import shlex
import sys
from pathlib import Path

NETWORKED = ("vastai", "ssh", "scp", "ssh-keyscan", "curl", "uv", "git")
ID = "54453954"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
ADAPTER_REPO = "Deco354/fc-em-adapters"


def vastai_raw(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)  # how vastai prints a --raw reply


def manifest(
    commit: str = "a" * 40,
    adapter_repo: str | None = ADAPTER_REPO,
    base: str = BASE,
    dirty: bool = False,
) -> dict:
    """The parts of a run's manifest.json (finetune/manifest.py) that up.sh --study reads."""
    run = {"study": "s", "run_id": "r", "arm": "a", "purpose": "plumbing", "pilot": False}
    if adapter_repo is not None:  # absent from manifests planned before schema 2
        run["adapter_repo"] = adapter_repo
    return {
        "schema_version": 2 if adapter_repo is not None else 1,
        "run": run,
        "code": {"git_commit": commit, "dirty": dirty, "diff_sha256": None},
        "model": {"base_model": base, "base_revision": BASE_REV},
    }


def write_study(study: Path, *runs: dict) -> Path:
    """A planned study: one <run>/manifest.json per manifest, two clean runs by default."""
    for i, doc in enumerate(runs or (manifest(), manifest())):
        (study / f"arm{i}--s0").mkdir(parents=True)
        (study / f"arm{i}--s0" / "manifest.json").write_text(json.dumps(doc, indent=1))
    return study


class Fakes:
    def __init__(self, home: Path):
        self.home = home
        self.dir = home / "fakes"
        self.dir.mkdir()
        (home / ".ssh").mkdir()
        self.state = home / ".config" / "fragile-compassion" / "vast-instance"
        self.state.parent.mkdir(parents=True)

    def vastai(self, subcommand: str, reply: str) -> None:
        """What `vastai <subcommand> ...` prints, e.g. vastai("show instance", ...)."""
        (self.dir / f"vastai {subcommand}").write_text(reply)

    def record_box(self, instance_id: str = ID) -> None:
        """The record up.sh leaves after renting: "<instance id> <commit>"."""
        self.state.write_text(f"{instance_id} {'0' * 40}\n")

    def calls(self) -> list[str]:
        path = self.dir / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def env(self, **extra: str) -> dict[str, str]:
        log = 'echo "$FUNCNAME $*" >>"$FAKE_DIR/calls"'
        env = {
            "HOME": str(self.home),
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "FAKE_DIR": str(self.dir),
            "BASH_FUNC_vastai%%": "() { "
            + log
            + '; local reply="$FAKE_DIR/vastai $1 $2"; if [ -f "$reply" ]; then cat "$reply";'
            ' else echo "fake vastai: no reply for: $*" >&2; return 1; fi; }',
            "BASH_FUNC_sleep%%": "() { :; }",
            "BASH_FUNC_python3%%": f'() {{ {shlex.quote(sys.executable)} "$@"; }}',
            **extra,
        }
        for name in NETWORKED[1:]:
            env[f"BASH_FUNC_{name}%%"] = (
                "() { " + log + '; echo "fake $FUNCNAME: offline" >&2; return 255; }'
            )
        return env
