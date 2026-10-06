#!/usr/bin/env bash
# Prepare a rented GPU box for `fc run`. Runs ON THE BOX, as root.
#
# scripts/vast/up.sh copies this file to /root/fc-box-setup.sh and starts it in the
# background as a login shell (`bash -l`), so it sees the same environment as an
# interactive `ssh vast-em`, the one `fc run` will use. Every step is idempotent: up.sh
# runs it again on the same box after a failure. It holds no secrets; up.sh writes the
# judge key into .env with --set-judge once setup has finished.
#
#   FC_REPO_URL=https://github.com/<owner>/<repo>.git FC_REF=<40-hex commit> bash -l fc-box-setup.sh
#   bash -l fc-box-setup.sh --check   # every pinned model resolves offline; public listeners
#   bash fc-box-setup.sh --set-judge < judge.env   # replace the judge key block in .env
#
# Progress: /root/fc-setup.log. Finished: /root/fc-setup.done. Failed: /root/fc-setup.failed.
set -Eeuo pipefail

DIR=${FC_DIR:-/root/fragile-compassion}
UV_VERSION=0.12.21
UV=/usr/local/bin/uv # this uv, not whichever one the image may put first on PATH
LOG=/root/fc-setup.log

# Every (repo, revision) in configs/models.yaml, base models included. "download" fetches
# the whole repo at that commit into the HF cache, which is what vLLM (base) and Inspect
# (adapters) read; "check" only resolves it from the cache, offline.
pinned_models() {
  local offline=0
  [ "$1" = check ] && offline=1
  HF_HUB_OFFLINE=$offline HF_HUB_DISABLE_PROGRESS_BARS=1 "$UV" run --no-sync python - "$1" <<'PY'
import sys

from huggingface_hub import snapshot_download

from fragile_compassion.config import expand_with_bases, load_text, parse_models_yaml

check = sys.argv[1] == "check"
pinned = []
for m in expand_with_bases(parse_models_yaml(load_text("configs/models.yaml"))):
    for repo, rev in ((m.base, m.base_revision), (m.adapter, m.adapter_revision)):
        if repo and (repo, rev) not in pinned:
            pinned.append((repo, rev))

missing = 0
for repo, rev in pinned:
    label = f"{repo}@{rev[:12]}"
    if not check:
        print(f"   {label}", flush=True)
        snapshot_download(repo, revision=rev)
        continue
    try:
        snapshot_download(repo, revision=rev)
        print(f"   cached   {label}")
    except Exception as exc:  # noqa: BLE001 - report every missing repo, not just the first
        missing += 1
        print(f"   MISSING  {label} ({type(exc).__name__})")
sys.exit(1 if missing else 0)
PY
}

# The box's .env in the current directory, mode 600; up.sh appends the judge key later.
# Each line is added only if it is missing, so rerunning on an older box adds new
# settings and keeps its key.
# - Inspect's auto-started vLLM server binds 0.0.0.0 with the well-known key `inspectai`
#   unless told otherwise. The key is per box and disposable: it shows in `ps` and in
#   Inspect's log of the vllm command.
# - vLLM's process groups use PyTorch's Gloo backend, which listens on the address the
#   container's hostname resolves to (its Docker-network address, 172.17.x.x), where
#   other containers on the same host may reach it. GLOO_SOCKET_IFNAME=lo moves those
#   listeners to 127.0.0.1. VLLM_HOST_IP does not: vLLM uses it only for the address it
#   advertises (checked on vLLM 0.28.0, 2026-10-06).
write_env() {
  local line
  if [ ! -f .env ]; then
    (
      umask 077
      echo "# Written by scripts/vast/box-setup.sh. up.sh adds the judge key block below." >.env
    )
  fi
  chmod 600 .env
  if ! grep -q '^VLLM_API_KEY=' .env; then
    echo "VLLM_API_KEY=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')" >>.env
  fi
  for line in 'VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}' 'VLLM_HOST_IP=127.0.0.1' \
    'GLOO_SOCKET_IFNAME=lo'; do
    grep -qxF "$line" .env || echo "$line" >>.env
  done
}

setup() {
  exec >>"$LOG" 2>&1
  echo $$ >/root/fc-setup.pid
  rm -f /root/fc-setup.done /root/fc-setup.failed
  trap 'echo "== error at line $LINENO: $BASH_COMMAND"' ERR
  trap 'rc=$?; [ "$rc" -eq 0 ] || { touch /root/fc-setup.failed; echo "== FAILED (exit $rc)"; }' EXIT
  echo "== fc box setup $(date -u +%FT%TZ) ref=${FC_REF:-unset}"

  : "${FC_REPO_URL:?FC_REPO_URL is required (up.sh sets it)}"
  [[ ${FC_REF:-} =~ ^[0-9a-f]{40}$ ]] || {
    echo "== FC_REF must be a 40-hex commit (up.sh resolves it), got '${FC_REF:-}'"
    exit 2
  }

  if ! command -v tmux >/dev/null; then
    echo "== installing tmux"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
  fi
  # Vast starts tmux on every SSH login unless this file exists. The README has you run
  # `tmux new -s fc` yourself, which then behaves the same in a terminal and in VS Code.
  touch /root/.no_auto_tmux

  if [ "$("$UV" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
    echo "== installing uv $UV_VERSION"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" |
      env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  fi

  if [ ! -d "$DIR/.git" ]; then
    git clone --quiet "$FC_REPO_URL" "$DIR" || {
      echo "== git clone $FC_REPO_URL failed. The repo must be public: the box has no GitHub credentials."
      exit 1
    }
  fi
  cd "$DIR"
  git fetch --quiet origin
  git checkout --quiet --detach "$FC_REF"
  echo "== commit $(git rev-parse HEAD)"

  write_env # before anything slow

  echo "== uv sync (about 10 GB of wheels)"
  "$UV" sync --group dev --extra vllm --locked

  # Otherwise the first `fc run` spends its first minutes downloading ~35 GB while the
  # GPU idles; Inspect 0.3.263 sets no vLLM start timeout, so it would just wait.
  echo "== model downloads (about 35 GB)"
  pinned_models download

  touch /root/fc-setup.done
  echo "== done $(date -u +%FT%TZ)"
}

check() {
  local rc=0
  cd "$DIR"
  echo "== pinned models in the HF cache this shell sees (HF_HOME=${HF_HOME:-unset, so ~/.cache/huggingface})"
  pinned_models check || rc=1
  echo "== uv on PATH: $(command -v uv || echo none) ($(uv --version 2>/dev/null || true))"
  if [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    echo "== Hugging Face token: present (written by up.sh from hf.env)"
  else
    echo "== Hugging Face token: none (anonymous downloads; gated datasets such as TAC's fail)"
  fi
  echo "== sockets listening on a non-loopback address (expect only sshd, port 22):"
  ss -ltnpH 2>/dev/null | awk '$4 !~ /^(127\.|\[::1\]:|\[::ffff:127\.)/ {print "   " $4, $6}' ||
    echo "   (ss is not installed; can't list them)"
  return "$rc"
}

# set_judge < judge.env: replaces the judge key block in the .env of the current
# directory with stdin, so a changed judge.env reaches a box that is already set up.
# The block sits between two marker lines. Also removed: the single marker line of
# earlier versions, and any other line setting a name that stdin sets. Every other line
# (the vLLM settings, the per-box key) stays as it was. Mode stays 600.
JUDGE_BEGIN="# >>> judge key, from judge.env (up.sh replaces this block on every run)"
JUDGE_END="# <<< judge key"
set_judge() {
  local new names
  new=$(cat)
  names=$(printf '%s\n' "$new" | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')
  (
    umask 077
    touch .env
    awk -v names="$names" -v begin="$JUDGE_BEGIN" -v end="$JUDGE_END" '
      BEGIN { n = split(names, list, "\n"); for (i = 1; i <= n; i++) drop[list[i]] = 1 }
      $0 == begin { inside = 1; next }
      $0 == end { inside = 0; next }
      inside { next }
      $0 == "# judge credential (appended by scripts/vast/up.sh)" { next }
      { name = $0; sub(/=.*/, "", name); if (name in drop) next }
      { lines[++count] = $0 }
      END {
        while (count > 0 && lines[count] == "") count--   # no trailing blank lines
        for (i = 1; i <= count; i++) print lines[i]
      }' .env >.env.next
    {
      echo
      echo "$JUDGE_BEGIN"
      printf '%s\n' "$new"
      echo "$JUDGE_END"
    } >>.env.next
    chmod 600 .env.next
    mv .env.next .env
  )
}

# Sourced (by the tests) it only defines the functions above.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    "") setup ;;
    --check) check ;;
    --set-judge)
      cd "$DIR"
      set_judge
      ;;
    *)
      echo "usage: $0 [--check | --set-judge < judge.env]" >&2
      exit 2
      ;;
  esac
fi
