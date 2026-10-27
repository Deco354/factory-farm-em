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


def vastai_raw(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)  # how vastai prints a --raw reply


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
