"""CI's training job runs on a pull request only when the PR changes a path in the
`changes` job's filter (.github/workflows/ci.yml). These tests keep that filter in step
with what the training suite depends on: every file the training tests import, directly
or through other modules (imports inside functions included), and the files the job
loads by path: the training envs, tools and configs, the lock files, and conftest.py.

conftest.py's own import of the CLI (for the `fc` fixture, which no training test uses)
isn't followed. The CLI imports nearly every module, so following it would put the whole
package in the filter, and all the training suite needs from those modules is that they
import, which the `test` job checks on every PR by loading the same conftest.py. The same
goes for the job's `fc train fetch-tokenizers` step: its own code is in finetune/, and
the light suite runs `fc train` through the CLI.

They read the workflow file itself: what GitHub does with it can't be run offline.
dorny/paths-filter matches with picomatch (`dot: true`). `pattern_regex` handles only the
part of that syntax the filter uses, `*` within a name and `**` as the last segment, and
fails on anything else, so the two can't disagree unnoticed."""

import ast
import re
from pathlib import Path

import yaml

from fragile_compassion.finetune.recipe import TRAINING_CONFIGS
from fragile_compassion.finetune.runner import TOOLS_DIR, TRAINING_ENVS, env_dir

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
FIRST_PARTY = ("fragile_compassion", "tests")


def filter_step():
    steps = yaml.safe_load(CI.read_text())["jobs"]["changes"]["steps"]
    (step,) = [s for s in steps if s.get("uses", "").startswith("dorny/paths-filter@")]
    return step


def pattern_regex(pattern):
    assert not re.search(r"[?\[\]{}()!\\]", pattern), f"{pattern}: syntax this test can't read"
    *head, last = pattern.split("/")
    assert all("**" not in s for s in head) and (last == "**" or "**" not in last), pattern
    parts = [re.escape(s).replace(r"\*", "[^/]*") for s in head]
    tail = ".+" if last == "**" else re.escape(last).replace(r"\*", "[^/]*")
    return re.compile("/".join([*parts, tail]) + r"\Z")


def unmatched(paths):
    patterns = yaml.safe_load(filter_step()["with"]["filters"])["training"]
    regexes = [pattern_regex(p) for p in patterns]
    rel = sorted(str(p.relative_to(ROOT)) for p in paths)
    return [p for p in rel if not any(r.match(p) for r in regexes)]


def module_file(name):
    base = ROOT / "src" if name.split(".")[0] == "fragile_compassion" else ROOT
    path = base.joinpath(*name.split("."))
    for f in (path.with_suffix(".py"), path / "__init__.py"):
        if f.is_file():
            return f
    return None  # a name imported from a module, not a module


def imported_files(path):
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: relative import, which this test doesn't follow"
            names += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    files = []
    for name in names:
        parts = name.split(".")
        if parts[0] in FIRST_PARTY:
            # Importing a.b.c runs a/__init__.py and a/b/__init__.py too.
            files += [module_file(".".join(parts[:i])) for i in range(1, len(parts) + 1)]
    return [f for f in files if f]


def import_closure(roots):
    seen, todo = set(), list(roots)
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo += imported_files(path)
    return seen


def test_every_file_the_training_tests_import_is_in_the_filter():
    reached = import_closure(ROOT.glob("tests/test_training_*.py"))
    assert ROOT / "src" / "fragile_compassion" / "finetune" / "tokens.py" in reached
    assert unmatched(reached) == []


def test_the_files_the_training_job_loads_by_path_are_in_the_filter():
    read = [ROOT / "pyproject.toml", ROOT / "uv.lock", CI, ROOT / "tests" / "conftest.py"]
    for env in TRAINING_ENVS:
        read += [env_dir(env) / "pyproject.toml", env_dir(env) / "uv.lock"]
    read += [*TOOLS_DIR.glob("*.py"), *TRAINING_CONFIGS.rglob("*.yaml")]
    assert all(p.is_file() for p in read) and len(read) > 9
    assert unmatched(read) == []


def test_a_change_to_only_docs_or_the_vast_scripts_skips_the_training_job():
    outside = [*ROOT.glob("docs/*.md"), ROOT / "README.md", *ROOT.glob("scripts/vast/*.sh")]
    assert len(outside) > 5
    assert len(unmatched(outside)) == len(outside)


def test_the_filter_action_is_pinned_to_a_commit_and_lists_no_files():
    step = filter_step()
    # A tag can be moved to other code; a full commit can't.
    assert re.fullmatch(r"dorny/paths-filter@[0-9a-f]{40}", step["uses"])
    # GHSA-7hc6-8hq5-9q2m: file names could inject shell commands through the file lists.
    assert "list-files" not in step["with"]
