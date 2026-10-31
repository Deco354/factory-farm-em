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
#
# Also from up.sh: FC_BOX_TRAINING=1 installs the training envs too (up.sh --study);
# FC_BOX_MODELS names the models to pre-download (see pinned_models).
#   bash fc-box-setup.sh --set-judge < judge.env   # replace the judge key block in .env
#   FC_BOX_TRAINING=1 FC_BOX_MODELS=... bash -l fc-box-setup.sh --ensure
#                       # on a box already set up: install and download what is missing
#
# Progress: /root/fc-setup.log. Finished: /root/fc-setup.done. Failed: /root/fc-setup.failed.
set -Eeuo pipefail

DIR=${FC_DIR:-/root/fragile-compassion}
UV_VERSION=0.12.21
UV=/usr/local/bin/uv # this uv, not whichever one the image may put first on PATH
LOG=/root/fc-setup.log
BOX_MODELS_RECORD=/root/fc-box-models # what setup downloaded, for a hand-run --check

# box_models: what this box pre-downloads. FC_BOX_MODELS when set (up.sh passes it), else
# what setup recorded, else configs/models.yaml.
box_models() {
  if [ -n "${FC_BOX_MODELS:-}" ]; then
    printf '%s\n' "$FC_BOX_MODELS"
  elif [ -s "$BOX_MODELS_RECORD" ]; then
    cat "$BOX_MODELS_RECORD"
  else
    echo configs/models.yaml
  fi
}

# Every (repo, revision) in box_models: space-separated models.yaml paths, whose models
# count base models included, and `repo@<40-hex>` pins (a training box needs only its
# base, which a models.yaml can't list alone); config.box_model_pins parses them.
# "download" fetches the whole repo at that commit into the HF cache, which is what vLLM
# (base), Inspect (adapters) and Axolotl read; "check" only resolves it from the cache,
# offline.
pinned_models() {
  local offline=0
  [ "$1" = check ] && offline=1
  FC_BOX_MODELS=$(box_models) HF_HUB_OFFLINE=$offline HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$UV" run --no-sync python - "$1" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

from fragile_compassion.config import box_model_pins

check = sys.argv[1] == "check"
pinned = box_model_pins(os.environ.get("FC_BOX_MODELS") or "configs/models.yaml")
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
  # Telemetry off: Axolotl sends PostHog telemetry by default, and the training data is
  # derived from copyrighted manuals. `fc train` also sets these in its subprocesses;
  # here they cover a hand-run `axolotl` too.
  for line in 'VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}' 'VLLM_HOST_IP=127.0.0.1' \
    'GLOO_SOCKET_IFNAME=lo' 'AXOLOTL_DO_NOT_TRACK=1' 'DO_NOT_TRACK=1' \
    'HF_HUB_DISABLE_TELEMETRY=1' 'WANDB_MODE=disabled'; do
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
  box_models >"$BOX_MODELS_RECORD"

  echo "== uv sync (about 10 GB of wheels)"
  "$UV" sync --group dev --extra vllm --locked

  training_envs

  # Otherwise the first `fc run` spends its first minutes downloading the models while the
  # GPU idles; Inspect 0.3.263 sets no vLLM start timeout, so it would just wait.
  echo "== model downloads"
  pinned_models download

  touch /root/fc-setup.done
  echo "== done $(date -u +%FT%TZ)"
}

# FC_BOX_TRAINING=1 (up.sh --study): both training envs, each its own uv project and lock.
training_envs() {
  local env
  [ "${FC_BOX_TRAINING:-0}" = 1 ] || return 0
  for env in axolotl-0.21.1 axolotl-e77a185; do
    echo "== uv sync training/envs/$env (several GB of CUDA wheels)"
    "$UV" sync --project "training/envs/$env" --locked
  done
}

# ensure: on a box whose setup has finished, what this up.sh run needs that setup didn't
# install: the training envs and FC_BOX_MODELS. Setup runs once per box, so without this a
# resumed eval box given a study would have neither. In the foreground, under up.sh. Both
# steps are idempotent: what is already there is only checked.
ensure() {
  cd "$DIR"
  training_envs
  echo "== model downloads (only what isn't cached yet)"
  pinned_models download
  add_to_record
}

# add_to_record: what box_models names, added to what the record already holds (each entry
# once), so a hand-run --check checks every model the box was given, not only the last
# run's.
add_to_record() {
  local all
  all=$({
    cat "$BOX_MODELS_RECORD" 2>/dev/null || true
    box_models
  } | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')
  printf '%s\n' "${all% }" >"$BOX_MODELS_RECORD"
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
    --ensure) ensure ;;
    --set-judge)
      cd "$DIR"
      set_judge
      ;;
    *)
      echo "usage: $0 [--check | --ensure | --set-judge < judge.env]" >&2
      exit 2
      ;;
  esac
fi
