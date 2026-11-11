"""Run a scripts/vast/ script offline: every networked command is a fake.

pytest-socket can't see a bash subprocess, so the script itself must be kept off the
network. Each command that could reach it (`NETWORKED`) is replaced by a bash function.
`vastai` answers from reply files a test writes. All the others fail the way a host
that can't be reached does (an SSH exit code, 255), so `ssh vast-em` reads as "can't
reach the box". Every call is appended to `calls`, and `sleep` returns at once.

A test that needs a box ssh can reach calls `Fakes.reachable_box()`: a directory then
stands in for the box. `ssh vast-em CMD` runs CMD there with bash, in the box's home, with
each "/root/" in CMD pointing into the stand-in for /root, and `pgrep` on the box reads
the process list `Fakes.processes` wrote.

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


# `ssh ... vast-em CMD`: offline unless the test made a box (FAKE_BOX), which then runs
# CMD. `-n` closes its stdin, as ssh's does; `box_stdin` sends it data.
_SSH = r"""() { %s
  if [ -z "${FAKE_BOX:-}" ]; then echo "fake $FUNCNAME: offline" >&2; return 255; fi
  local closed=0 cmd
  while [ $# -gt 0 ] && [ "$1" != vast-em ]; do
    if [ "$1" = -n ]; then closed=1; fi
    shift
  done
  [ $# -gt 1 ] || return 0
  shift
  cmd=$*
  cmd=${cmd//\/root\//$FAKE_BOX/root/}
  if [ "$closed" = 1 ]; then
    (cd "$FAKE_BOX/root" && bash -c "$cmd" </dev/null)
  else
    (cd "$FAKE_BOX/root" && bash -c "$cmd")
  fi
}"""

# `pgrep -af PATTERN` on the box: the lines of its process list ("<pid> <command line>")
# that PATTERN matches, with grep -E, the regex dialect pgrep uses. Exits 1 when none do,
# as pgrep does; a box without a list runs nothing. `Fakes.pgrep_fails` makes it fail as
# pgrep does when it can't read the process list (exit 3).
_PGREP = r"""() { %s
  local pattern list="${FAKE_BOX:-/nonexistent}/processes"
  if [ -f "${FAKE_BOX:-/nonexistent}/pgrep-fails" ]; then
    echo "pgrep: Cannot get process list" >&2
    return 3
  fi
  for pattern; do :; done
  [ -f "$list" ] || return 1
  grep -E -- "$pattern" "$list"
}"""


class Fakes:
    def __init__(self, home: Path):
        self.home = home
        self.dir = home / "fakes"
        self.dir.mkdir()
        (home / ".ssh").mkdir()
        self.state = home / ".config" / "fragile-compassion" / "vast-instance"
        self.state.parent.mkdir(parents=True)
        self.box: Path | None = None

    def vastai(self, subcommand: str, reply: str) -> None:
        """What `vastai <subcommand> ...` prints, e.g. vastai("show instance", ...)."""
        (self.dir / f"vastai {subcommand}").write_text(reply)

    def record_box(self, instance_id: str = ID, commit: str = "0" * 40) -> None:
        """The record up.sh leaves after renting: "<instance id> <commit>"."""
        self.state.write_text(f"{instance_id} {commit}\n")

    def reachable_box(self) -> Path:
        """A box `ssh vast-em` reaches, empty and idle; returns the stand-in for its /root."""
        self.box = self.home / "box"
        (self.box / "root").mkdir(parents=True)
        return self.box / "root"

    def processes(self, *lines: str) -> None:
        """What runs on the box, as `pgrep -a` prints it: "<pid> <command line>" each."""
        assert self.box is not None, "reachable_box() first"
        (self.box / "processes").write_text("".join(f"{line}\n" for line in lines))

    def pgrep_fails(self) -> None:
        """pgrep on the box can't read the process list, and says so."""
        assert self.box is not None, "reachable_box() first"
        (self.box / "pgrep-fails").touch()

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
            "BASH_FUNC_pgrep%%": _PGREP % log,
            **({"FAKE_BOX": str(self.box)} if self.box else {}),
            **extra,
        }
        for name in NETWORKED[1:]:
            env[f"BASH_FUNC_{name}%%"] = (
                "() { " + log + '; echo "fake $FUNCNAME: offline" >&2; return 255; }'
            )
        env["BASH_FUNC_ssh%%"] = _SSH % log
        return env
