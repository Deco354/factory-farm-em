"""Hugging Face checks for training: the upload token's rule, and secret values on a box.

`write_token_problem` is scripts/vast/common.sh's `hf_write_token_problem` in Python, with
the same messages: `up.sh --study` checks the upload token before renting, and `fc train
run` checks it again on the box before training, since a box can stay rented for days and
a fine-grained token can expire or be revoked. The two are tested on the same replies
(tests/hf_whoami.py), so they can't drift apart.

The calls that reach Hugging Face are kept thin; the rules are pure.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

from fragile_compassion.finetune.runner import SECRET_NAME

# Where up.sh --study puts the upload token on a box (common.sh `upload_token_to_box`).
BOX_UPLOAD_TOKEN = Path("/root/.config/fc/hf-write-token")
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
MIN_SECRET_LENGTH = 8  # shorter values ("1", "true") would match everywhere


def write_token_problem(raw: str, repo: str) -> str | None:
    """What's wrong with a whoami-v2 reply for the upload token, or None if it may write
    to `repo` and do nothing else beyond reading."""
    try:
        d = json.loads(raw)
    except ValueError:
        return "expected JSON from Hugging Face, got: " + (raw.strip()[:200] or "nothing")
    if not isinstance(d, Mapping):
        return "expected JSON from Hugging Face, got: " + raw.strip()[:200]
    if "error" in d:
        return f"Hugging Face rejected the token: {d['error']}"
    token = (d.get("auth") or {}).get("accessToken") or {}
    if token.get("role") != "fineGrained":
        return (
            f"the upload token must be fine-grained, scoped to {repo} (role: {token.get('role')})"
        )
    fine = token.get("fineGrained") or {}
    beyond = [f"global: {p}" for p in fine.get("global") or [] if not p.endswith(".read")]
    can_write_repo = False
    for scope in fine.get("scoped") or []:
        entity = scope.get("entity") or {}
        is_repo = entity.get("type") == "model" and entity.get("name") == repo
        for p in scope.get("permissions") or []:
            if p.endswith(".read"):
                continue
            if p == "repo.write" and is_repo:
                can_write_repo = True
                continue
            beyond.append(f"{entity.get('type')} {entity.get('name')}: {p}")
    if beyond:
        return f"the token can do more than write to {repo}: {'; '.join(sorted(beyond))}"
    if not can_write_repo:
        return f"the token has no repo.write permission scoped to {repo}"
    return None


def whoami_raw(token: str, *, timeout: float = 30) -> str:
    """The whoami-v2 reply for `token`, as text. Hugging Face answers a bad token with a
    JSON error body, which is returned like a success so the rule can say what it was."""
    req = urllib.request.Request(WHOAMI_URL, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.read().decode("utf-8", "replace")


def read_token(path: Path) -> str:
    """A token file's value: the bare token, as up.sh writes it on a box."""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if not token:
        raise FileNotFoundError(
            f"no upload token at {path}; rent the box with scripts/vast/up.sh --study "
            "runs/<study>, which puts it there"
        )
    return token


def check_upload_token(repo: str, token_file: Path = BOX_UPLOAD_TOKEN) -> None:
    """Raise unless the token in `token_file` can still write to `repo`, and nothing more."""
    problem = write_token_problem(whoami_raw(read_token(token_file)), repo)
    if problem:
        raise PermissionError(f"{token_file}: {problem}")


# ------------------------------------------------------------------ secrets on a box


def hf_home(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def box_secret_files(environ: Mapping[str, str], repo_root: Path) -> list[Path]:
    """Every file on a box that holds a secret: the upload token, Hugging Face's own
    token file (the read token) and the repo's .env (the judge key, vLLM's API key)."""
    return [BOX_UPLOAD_TOKEN, hf_home(environ) / "token", repo_root / ".env"]


def secret_values(texts: Iterable[str]) -> set[str]:
    """The secret values in token files and .env files: a bare token, or the value of
    every `NAME=value` line whose name looks like a credential."""
    values: set[str] = set()
    for text in texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) == 1 and "=" not in lines[0]:
            values.add(lines[0])
            continue
        for line in lines:
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.removeprefix("export ").partition("=")
            value = value.strip().strip("'\"")
            if SECRET_NAME.search(name):
                values.add(value)
    return {v for v in values if len(v) >= MIN_SECRET_LENGTH}


def read_secret_values(files: Iterable[Path]) -> set[str]:
    texts = []
    for f in files:
        try:
            texts.append(f.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            continue
    return secret_values(texts)


def files_with_secrets(paths: Iterable[Path], values: set[str]) -> list[Path]:
    """The files among `paths` (directories searched whole) that contain any of `values`.
    Only names come back, never the values."""
    if not values:
        return []
    needles = [v.encode("utf-8") for v in values]
    hits = []
    for root in paths:
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for f in files:
            if not f.is_file() or f.is_symlink():
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if any(n in data for n in needles):
                hits.append(f)
    return hits
