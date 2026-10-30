#!/usr/bin/env bash
# Rent a Vast.ai GPU box and get it ready for `fc run`. Runs on your laptop (macOS or Linux).
#
#   scripts/vast/up.sh [--eval FILE] [--models FILE] [--judge FILE] [--dry-run] [--yes]
#   scripts/vast/up.sh --study runs/<study>    # a training box for a planned study
#   scripts/vast/up.sh --push runs/<study>     # copy a re-planned study to the recorded box
#
# First checks everything it can for free: tools, SSH config, that the commit is on
# GitHub, the run's config files, and one real judge call made with only the box's key.
# Then it sizes the box from the models the run uses and shows the cheapest matching
# offer, counting GPU time and setup downloads, rents it once you confirm, and points
# `ssh vast-em` at it. It then runs box-setup.sh there (uv sync, the run's pinned models)
# and copies the judge key and the run's config files over. If it fails or you press
# Ctrl-C after renting, run it again: it picks up the recorded box instead of renting a
# second one, and downloads whatever models the run now needs. See docs/vast.md.
#
#   --eval FILE   the eval profile the box will run (configs/eval.yaml). The box
#                 pre-downloads the profile's models and the preflight calls its judge.
#   --models FILE, --judge FILE
#                 instead of the profile's own, as with `fc run`. All three files are
#                 copied to the same paths in the box's checkout, edits included, so
#                 they must be in this repo.
#   --dry-run   run every check and the offer search, then stop without renting
#   --yes       answer yes to every question (for scripted use)
#   --study DIR set the box up to train the study `fc train plan` wrote to DIR (under
#               runs/): install the training envs, pre-download the study's base model,
#               check the upload token against the study's adapter_repo (which must be
#               private), and copy the study over once setup is done. On a recorded box,
#               install and download what it lacks. The study must have been planned,
#               clean, at the commit the box runs.
#   --push DIR  copy a re-planned study (a directory under runs/, without its
#               _candidates/) to the recorded box, then stop. The same commit rule.
#
# Settings, as environment variables or FC_NAME=value lines in a settings file
# (~/.config/fragile-compassion/vast.env, or the file FC_VAST_SETTINGS names). The
# environment wins over the file. Defaults:
#   FC_REF          commit for the box to check out   (HEAD; must be on GitHub)
#   FC_REPO_URL     repo the box clones               (origin, as https)
#   FC_JUDGE_ENV    judge key file copied to the box  (~/.config/fragile-compassion/judge.env)
#   FC_HF_ENV       optional read-only Hugging Face token for the box
#                                                     (~/.config/fragile-compassion/hf.env)
#   FC_VAST_SSH_KEY private key for the box           (~/.ssh/vastai)
#   FC_VAST_IMAGE   Docker image                      (see IMAGE below)
#   FC_VAST_DISK    disk in GB                        (sized from the run's models; box_disk)
#   FC_VAST_QUERY   `vastai search offers` filter     (see QUERY below; replaces it all)
#   FC_VAST_QUERY_EXTRA clauses added after the query (e.g. 'reliability>0.99 inet_down=any')
#   FC_VAST_HOURS   GPU hours used to rank offers     (1; raise it for long runs)
#   FC_VAST_MAX_DPH most $/hr it will ever rent at    (3.00 for GPU + disk; applies even with --yes)
#   FC_BOX_MODELS   more models to pre-download: models.yaml paths and repo@<commit> pins,
#                   space-separated                   (none)
#   FC_HF_WRITE_ENV upload token for a --study box, may write only to the study's
#                   adapter_repo                      (~/.config/fragile-compassion/hf-write.env)
set -euo pipefail

# shellcheck source=scripts/vast/common.sh
. "$(dirname "$0")/common.sh"

# --------------------------------------------------------------------- settings file
# One FC_NAME=value per line; '#' starts a comment line; one pair of surrounding quotes
# is removed; nothing is expanded or executed. Unknown names fail, so a typo can't be
# silently ignored.
SETTINGS=${FC_VAST_SETTINGS:-$STATE_DIR/vast.env}
KNOWN_SETTINGS=" FC_REF FC_REPO_URL FC_JUDGE_ENV FC_HF_ENV FC_VAST_SSH_KEY FC_VAST_IMAGE FC_VAST_DISK FC_VAST_QUERY FC_VAST_QUERY_EXTRA FC_VAST_HOURS FC_VAST_MAX_DPH FC_BOX_MODELS FC_HF_WRITE_ENV "
# Replaced by --study, which reads both from the study. Refused, not ignored: a value left
# in a shell or a settings file would otherwise look as if it still did something.
RETIRED_SETTINGS=" FC_BOX_TRAINING FC_ADAPTER_REPO "
retired() {
  die "$1 is gone: scripts/vast/up.sh --study runs/<study> sets up a training box from a study planned with fc train plan, its adapter repo included (docs/training.md). Remove $1 $2."
}
if [ -f "$SETTINGS" ]; then
  from_file="" from_env="" lineno=0
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))
    line=${line%$'\r'}
    line=${line#"${line%%[![:space:]]*}"} # leading whitespace
    case $line in '' | '#'*) continue ;; esac
    name=${line%%=*}
    value=${line#*=}
    case $line in *=*) ;; *) name="" ;; esac
    case $RETIRED_SETTINGS in *" $name "*) retired "$name" "from $SETTINGS line $lineno" ;; esac
    case $KNOWN_SETTINGS in
      *" $name "*) ;;
      *) die "$SETTINGS line $lineno: expected FC_NAME=value, with FC_NAME one of:$KNOWN_SETTINGS" ;;
    esac
    case $value in
      \"*\") value=${value#\"} value=${value%\"} ;;
      \'*\') value=${value#\'} value=${value%\'} ;;
    esac
    if [ -n "${!name+set}" ]; then
      from_env="$from_env $name"
    else
      printf -v "$name" '%s' "$value"
      from_file="$from_file $name"
    fi
  done <"$SETTINGS"
  [ -z "$from_file" ] || say "settings from $SETTINGS:$from_file"
  [ -z "$from_env" ] || say "set in your environment, so not taken from $SETTINGS:$from_env"
fi
for name in $RETIRED_SETTINGS; do
  [ -z "${!name+set}" ] || retired "$name" "from your environment (unset $name)"
done

JUDGE_ENV=${FC_JUDGE_ENV:-$STATE_DIR/judge.env}
# Optional: a fine-grained, read-only Hugging Face token for the box (docs/vast.md,
# one-time setup step 7). It authenticates the box's downloads, so fewer rate limits,
# and gated datasets (TAC's, ANIMA's) need it. Read only from this file, never your shell's
# HF_TOKEN; checked before renting; sent to the box over SSH stdin, never on a command
# line.
HF_ENV=${FC_HF_ENV:-$STATE_DIR/hf.env}
# token_file_value FILE: the token in FILE, which must hold exactly one HF_TOKEN=<token>
# line (one pair of surrounding quotes removed). Makes FILE private.
token_file_value() {
  local value
  [ "$(grep -c '^HF_TOKEN=.' "$1")" = 1 ] || die "$1 must hold exactly one HF_TOKEN=<token> line"
  value=$(sed -n 's/^HF_TOKEN=//p' "$1")
  value=${value%$'\r'}
  case $value in
    \"*\") value=${value#\"} value=${value%\"} ;;
    \'*\') value=${value#\'} value=${value%\'} ;;
  esac
  chmod 600 "$1"
  printf '%s' "$value"
}
HF_TOKEN_VALUE=""
if [ -f "$HF_ENV" ]; then
  HF_TOKEN_VALUE=$(token_file_value "$HF_ENV")
fi
# For `fc train upload` on a --study box (docs/vast.md, one-time setup step 8): a
# fine-grained token that may write to the study's adapter repo and nothing else, checked
# before renting. On the box it goes only to /root/.config/fc/hf-write-token (mode 600):
# never into .env, Hugging Face's own token file or a training process, so only the upload
# reads it. Without --study it is not read, and a box keeps no copy.
HF_WRITE_ENV=${FC_HF_WRITE_ENV:-$STATE_DIR/hf-write.env}
KEY=${FC_VAST_SSH_KEY:-$HOME/.ssh/vastai}
# Pinned by date: the matching `cuda-13.0.3-auto` tag moves. CUDA 13.0 matches the
# lockfile's torch/vLLM runtime. SSH launch mode skips the image's own entrypoint, so
# none of its web services (Instance Portal, Jupyter, Syncthing) start.
IMAGE=${FC_VAST_IMAGE:-vastai/base-image:cuda-13.0.3-cudnn-devel-ubuntu24.04-2026-09-07}
# Without FC_VAST_DISK, sized from the models the box pre-downloads (box_disk), once they
# are known.
DISK=${FC_VAST_DISK:-}
case ${DISK:-1} in *[!0-9]*) die "FC_VAST_DISK must be a whole number of GB, got '$DISK'" ;; esac
# Offers are ranked by GPU price x FC_VAST_HOURS + download price x DOWNLOAD_GB: hosts
# charge 0-40 $/TB for downloads, which the hourly price leaves out, and a new box
# downloads its models and wheels before it is ready (box_disk).
HOURS=${FC_VAST_HOURS:-1}
# A cap on the hourly price, outside FC_VAST_QUERY so overriding the query keeps it. It
# is sent to Vast and checked again on what comes back.
MAX_DPH=${FC_VAST_MAX_DPH:-3.00}
for setting in "FC_VAST_HOURS=$HOURS" "FC_VAST_MAX_DPH=$MAX_DPH"; do
  case ${setting#*=} in
    '' | .* | *. | *[!0-9.]* | *.*.*) die "${setting%%=*} must be a number, got '${setting#*=}'" ;;
  esac
done

DRY=0 PUSH="" STUDY="" EVAL_FILE="" MODELS_FILE="" JUDGE_FILE=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --yes | -y) YES=1 ;;
    --push | --study)
      [ $# -ge 2 ] || die "$1 needs a directory under runs/"
      if [ "$1" = --push ]; then PUSH=$2; else STUDY=$2; fi
      shift
      ;;
    --eval | --models | --judge)
      [ $# -ge 2 ] || die "$1 needs a file"
      file=$2
      case $file in /*) ;; *) file=$PWD/$file ;; esac # up.sh moves to the repo later
      case $1 in
        --eval) EVAL_FILE=$file ;;
        --models) MODELS_FILE=$file ;;
        *) JUDGE_FILE=$file ;;
      esac
      shift
      ;;
    -h | --help)
      awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0" # the header comment

      exit 0
      ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

# --------------------------------------------------------------------- studies
RUNS_DIR=$(cd "$REPO" && pwd -P)/runs
# study_dir FLAG DIR: DIR resolved, which must be a directory under runs/, where
# `fc train plan` writes studies and from which down.sh copies them back.
study_dir() {
  local dir
  dir=$(cd "$2" 2>/dev/null && pwd -P) || die "$1: no directory $2"
  case $dir/ in
    "$RUNS_DIR"/?*/) printf '%s\n' "$dir" ;;
    *) die "$1 takes a directory under $RUNS_DIR, got $2" ;;
  esac
}
# push_study DIR: a `fc train plan` bundle to the box, without its _candidates/. Training
# data is copied only like this, one study at a time: the host can read the box's disk.
push_study() {
  local rel=${1#"$RUNS_DIR"/}
  say "copying runs/$rel to the box, without _candidates/"
  # COPYFILE_DISABLE: macOS tar would otherwise add AppleDouble ._ files.
  COPYFILE_DISABLE=1 tar -C "$RUNS_DIR" --exclude="$rel/_candidates" -cf - "$rel" |
    box_stdin 'mkdir -p /root/fragile-compassion/runs && tar -C /root/fragile-compassion/runs -xf -'
  say "on the box: /root/fragile-compassion/runs/$rel"
}

# --push: a re-planned study to the box up.sh rented, then stop. A study trains on the
# commit it was planned at, as --study checks too: its manifests record that commit, and
# would otherwise describe other code than the box runs.
if [ -n "$PUSH" ]; then
  [ -z "$STUDY" ] || die "--push and --study don't go together: --study copies its study to the box itself"
  push_dir=$(study_dir --push "$PUSH")
  [ -f "$STATE" ] || die "--push: no box recorded in $STATE; rent one with up.sh first"
  read -r _ box_ref <"$STATE" || true
  settings=$(study_settings "$push_dir") || die "not copying $PUSH to the box (above)"
  commit=${settings##* }
  [ "$commit" = "${box_ref:-}" ] ||
    die "--push: $PUSH was planned at commit ${commit:0:12}, but the box runs ${box_ref:0:12}. Plan it again at that commit, or rent a new box for it with down.sh, then up.sh --study $PUSH"
  box 'test -d /root/fragile-compassion/.git' ||
    die "--push: can't reach the box with ssh vast-em, or box-setup.sh hasn't cloned the repo yet"
  push_study "$push_dir"
  exit 0
fi

# --study: a training box. The study's manifests say what it needs, so nothing about
# training is a setting: the base model to pre-download, the repo adapters upload to, and
# the commit the study was planned at.
BOX_TRAINING=0 STUDY_DIR="" STUDY_LABEL="" STUDY_COMMIT="" STUDY_PIN="" ADAPTER_REPO=""
HF_WRITE_VALUE=""
if [ -n "$STUDY" ]; then
  STUDY_DIR=$(study_dir --study "$STUDY")
  STUDY_LABEL=runs/${STUDY_DIR#"$RUNS_DIR"/}
  settings=$(study_settings "$STUDY_DIR") || die "can't set up a box for $STUDY (above)"
  read -r STUDY_PIN ADAPTER_REPO STUDY_COMMIT <<<"$settings"
  [[ $ADAPTER_REPO =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "$STUDY_LABEL: adapter_repo must be a Hugging Face repo id like org/name, got '$ADAPTER_REPO'"
  BOX_TRAINING=1
  [ -f "$HF_WRITE_ENV" ] ||
    die "no upload token at $HF_WRITE_ENV: a training box needs one that may write to $ADAPTER_REPO, or its adapters can't be uploaded after the GPU hours are paid (docs/vast.md one-time setup step 8; FC_HF_WRITE_ENV names another file)"
  HF_WRITE_VALUE=$(token_file_value "$HF_WRITE_ENV")
fi
case ${FC_BOX_MODELS:-} in
  *[!A-Za-z0-9._/@\ -]*)
    die "FC_BOX_MODELS takes models.yaml paths and repo@<commit> pins, separated by spaces; got '$FC_BOX_MODELS'"
    ;;
esac

ID=""
DPH="?"
billing_traps # from here on, any exit with a box rented says it is still billing

# --------------------------------------------------------------------- 1. local checks
for tool in vastai python3 ssh ssh-keyscan scp git uv curl; do
  command -v "$tool" >/dev/null || die "$tool not found on PATH (docs/vast.md, one-time setup)"
done
[ -f "$KEY" ] && [ -f "$KEY.pub" ] || die "no SSH key pair at $KEY and $KEY.pub (one-time setup step 4, or set FC_VAST_SSH_KEY)"
[ -f "$JUDGE_ENV" ] || die "no judge key file at $JUDGE_ENV (one-time setup step 6)"
grep -Eq '^[A-Za-z_][A-Za-z0-9_]*=.' "$JUDGE_ENV" || die "$JUDGE_ENV has no NAME=value line"
chmod 600 "$JUDGE_ENV"
grep -Eiq '^[[:space:]]*include[[:space:]].*vast-em\.conf' "$HOME/.ssh/config" 2>/dev/null ||
  die "add this as the first line of ~/.ssh/config:  Include ~/.ssh/vast-em.conf"

cd "$REPO"
REPO_P=$(pwd -P)

# --------------------------------------------------------------------- what the box runs
# run_files: the run's config files, resolved as `fc run` resolves them (fc box-needs), and
# every model the box pre-downloads, as repo@commit pins: the eval profile's models (not
# for a study box given no --eval or --models), the study's base model and FC_BOX_MODELS.
# Pins, not file names, so the box downloads what this laptop's files say, edits
# included. Sets EVAL_REL, MODELS_REL and JUDGE_REL (paths in the repo) and BOX_MODELS.
EVAL_REL="" MODELS_REL="" JUDGE_REL="" BOX_MODELS=""
run_files() {
  local args=(--eval "${EVAL_FILE:-$REPO_P/configs/eval.yaml}") extra needs key value rel
  [ -z "$MODELS_FILE" ] || args+=(--models "$MODELS_FILE")
  [ -z "$JUDGE_FILE" ] || args+=(--judge "$JUDGE_FILE")
  extra="$STUDY_PIN ${FC_BOX_MODELS:-}"
  [ -z "${extra// /}" ] || args+=(--extra "$extra")
  if [ -n "$STUDY" ] && [ -z "$EVAL_FILE$MODELS_FILE" ]; then
    args+=(--no-config-models)
  fi
  needs=$(uv run --quiet --project "$REPO" fc box-needs "${args[@]}") ||
    die "can't set up a box for these config files (above)"
  while read -r key value; do
    if [ "$key" = pin ]; then
      BOX_MODELS="${BOX_MODELS:+$BOX_MODELS }$value"
      continue
    fi
    rel=${value#"$REPO_P"/}
    [ "$rel" != "$value" ] ||
      die "the $key file $value is outside $REPO_P. up.sh copies config files to the same path in the box's checkout, so keep them in the repo."
    case $rel in
      *[!A-Za-z0-9._/-]*) die "the $key file $rel: up.sh copies only paths of letters, digits and ._/-" ;;
    esac
    case $key in
      eval) EVAL_REL=$rel ;;
      models) MODELS_REL=$rel ;;
      *) JUDGE_REL=$rel ;;
    esac
  done <<<"$needs"
  # It reaches the box inside single quotes, so only characters that need no quoting.
  case $BOX_MODELS in
    '' | *[!A-Za-z0-9._/@\ -]*) die "unexpected models to pre-download: '$BOX_MODELS'" ;;
  esac
}

# judge_preflight: one real call to the run's judge, in an environment holding nothing but
# $JUDGE_ENV (Inspect's get_model() does not read .env files). A wrong, revoked or
# unfunded key fails here instead of stalling `fc run` on a rented GPU.
judge_preflight() {
  local tmp judge_ok=1
  say "judge check: one call to the $JUDGE_REL judge, using only $JUDGE_ENV"
  tmp=$(mktemp -d)
  (cd "$tmp" && env -i HOME="$HOME" PATH="$PATH" TERM="${TERM:-dumb}" \
    uv run --quiet --project "$REPO" --env-file "$JUDGE_ENV" \
    python - "$REPO_P/$JUDGE_REL" <<'PY'
import asyncio
import sys

from inspect_ai.model import GenerateConfig

from fragile_compassion.config import load_text, parse_judge_yaml
from fragile_compassion.judge.scorer import judge_model


async def main() -> int:
    judge = parse_judge_yaml(load_text(sys.argv[1]))
    try:
        model = judge_model(judge.model, judge.temperature, judge.reasoning_effort, judge.max_tokens)
        # Bounded: with no max_retries set, Inspect retries a failing judge forever.
        out = await model.generate(
            "Reply with the number 7 and nothing else.",
            config=GenerateConfig(max_retries=2, timeout=60),
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "don't rent yet"
        print(f"   {judge.model}: {type(exc).__name__}: {exc}"[:800], file=sys.stderr)
        return 1
    reply = out.completion.strip()
    print(f"   {judge.model} replied {reply!r}")
    return 0 if reply and not out.error else 1


sys.exit(asyncio.run(main()))
PY
  ) || judge_ok=0
  rm -rf "$tmp"
  [ "$judge_ok" = 1 ] || die "judge check failed; fix $JUDGE_ENV before renting (docs/vast.md troubleshooting)"
}

# hf_get URL: a GET from Hugging Face, with the read-only hf.env token if there is one,
# sent as a header on stdin, never on a command line.
hf_get() {
  if [ -n "$HF_TOKEN_VALUE" ]; then
    printf 'Authorization: Bearer %s\n' "$HF_TOKEN_VALUE" | curl -sS --max-time 20 -H @- "$1" 2>&1
  else
    curl -sS --max-time 20 "$1" 2>&1
  fi
}

if [ -f "$STATE" ] && [ "$DRY" = 0 ]; then
  # ----------------------------------------------------------------- resume a recorded box
  read -r ID REF <"$STATE" || true
  [ -n "$ID" ] && [ -n "${REF:-}" ] || die "$STATE is malformed; check https://cloud.vast.ai/instances/ and delete it"
  say "resuming instance $ID, commit ${REF:0:12} (recorded in $STATE)"
  [ -z "$STUDY" ] || [ "$STUDY_COMMIT" = "$REF" ] ||
    die "$STUDY_LABEL was planned at commit ${STUDY_COMMIT:0:12}, but this box runs ${REF:0:12}. Plan the study again at ${REF:0:12}, or rent a new box for it: scripts/vast/down.sh, then up.sh --study $STUDY_LABEL"
  if instance_gone "$(instance_row "$ID")"; then
    rm -f "$STATE"
    gone_id=$ID
    ID=""
    die "instance $gone_id no longer exists on Vast; removed the record. Run up.sh again to rent a new box."
  fi
  REPO_URL=${FC_REPO_URL:-$(git remote get-url origin)}
else
  # ----------------------------------------------------------------- 2. the commit to run
  git fetch --quiet origin || die "git fetch origin failed"
  REF=$(git rev-parse --verify --quiet "${FC_REF:-HEAD}^{commit}") || die "FC_REF=${FC_REF:-} is not a commit in this repo"
  [ -n "$(git branch -r --contains "$REF" 2>/dev/null)" ] ||
    die "commit ${REF:0:12} is not on GitHub yet. Push it first: the box clones from there."
  [ -z "$STUDY" ] || [ "$STUDY_COMMIT" = "$REF" ] ||
    die "$STUDY_LABEL was planned at commit ${STUDY_COMMIT:0:12}, but the box would check out ${REF:0:12} (${FC_REF:-HEAD}). Plan the study again: fc train plan --replace"
  if [ -z "${FC_REF:-}" ] && [ -n "$(git status --porcelain)" ]; then
    warn "you have uncommitted changes; the box gets commit ${REF:0:12} without them:"
    git status --short | head -n 10 >&2
    confirm "Continue anyway?" || exit 1
  fi
  REPO_URL=${FC_REPO_URL:-$(git remote get-url origin)}

  # ----------------------------------------------------------------- 3. config, judge
  run_files
  judge_preflight
fi

# --------------------------------------------------------------------- Hugging Face token
# On every run, resumes and dry runs included: one free whoami call, the token sent as a
# header on stdin. A token that can do more than read never goes on a rented box.
if [ -n "$HF_TOKEN_VALUE" ]; then
  say "Hugging Face token check: $HF_ENV"
  hf_reply=$(printf 'Authorization: Bearer %s\n' "$HF_TOKEN_VALUE" |
    curl -sS --max-time 20 -H @- https://huggingface.co/api/whoami-v2 2>&1) ||
    die "could not reach Hugging Face to check the token: ${hf_reply:0:200}"
  printf '%s' "$hf_reply" | hf_token_problem ||
    die "not putting the token in $HF_ENV on a rented box (above); see docs/vast.md one-time setup step 7"
  echo "   read-only: OK"
fi
if [ -n "$HF_WRITE_VALUE" ]; then
  say "upload token check: $HF_WRITE_ENV may write to $ADAPTER_REPO and nothing else"
  hf_reply=$(printf 'Authorization: Bearer %s\n' "$HF_WRITE_VALUE" |
    curl -sS --max-time 20 -H @- https://huggingface.co/api/whoami-v2 2>&1) ||
    die "could not reach Hugging Face to check the token: ${hf_reply:0:200}"
  printf '%s' "$hf_reply" | hf_write_token_problem "$ADAPTER_REPO" ||
    die "not putting the token in $HF_WRITE_ENV on a rented box (above); see docs/vast.md one-time setup step 8"
  echo "   writes only $ADAPTER_REPO: OK"
  # The token's scope says nothing about the repo itself, so ask for the repo too.
  hf_reply=$(printf 'Authorization: Bearer %s\n' "$HF_WRITE_VALUE" |
    curl -sS --max-time 20 -H @- "https://huggingface.co/api/models/$ADAPTER_REPO" 2>&1) ||
    die "could not reach Hugging Face to check $ADAPTER_REPO: ${hf_reply:0:200}"
  printf '%s' "$hf_reply" | adapter_repo_problem "$ADAPTER_REPO" ||
    die "not setting up a box whose adapters could not be uploaded (above); see docs/vast.md one-time setup step 8"
  echo "   $ADAPTER_REPO exists and is private: OK"
fi

case $REPO_URL in
  git@github.com:*) REPO_URL=https://github.com/${REPO_URL#git@github.com:} ;;
  ssh://git@github.com/*) REPO_URL=https://github.com/${REPO_URL#ssh://git@github.com/} ;;
esac
case $REPO_URL in
  https://*) ;;
  *) die "can't clone $REPO_URL without credentials; set FC_REPO_URL to its https URL" ;;
esac

if [ -z "$ID" ]; then
  # ----------------------------------------------------------------- 4. Vast account
  listing=$(vastai show instances --raw 2>&1 || true)
  others=$(printf '%s' "$listing" | json '
if not isinstance(d, list):
    sys.exit("cannot list instances: %s" % d)
label = sys.argv[2]
print(" ".join("%s(%s)" % (i["id"], i.get("actual_status")) for i in d if i.get("label") == label))' "$LABEL") ||
    die "vastai is not logged in or its key lacks instance_read. Run: vastai set api-key <restricted key>"
  if [ -n "$others" ]; then
    warn "instances labelled $LABEL already exist on the account, and are billing: $others"
    warn "(destroy stray ones with: vastai destroy instance <id> -y, or scripts/vast/down.sh)"
    [ "$DRY" = 1 ] || confirm "Rent another box anyway?" || exit 1
  fi

  # ----------------------------------------------------------------- 5. size, offer, rent
  # The disk and the downloads offers are ranked on, from the pinned repos' sizes. One
  # free API call per repo; a repo or commit Hugging Face doesn't have stops here.
  model_bytes=0
  for pin in $BOX_MODELS; do
    reply=$(hf_get "https://huggingface.co/api/models/${pin%@*}/revision/${pin##*@}?blobs=true") ||
      die "could not reach Hugging Face to size $pin: ${reply:0:200}"
    bytes=$(printf '%s' "$reply" | pin_bytes "$pin") ||
      die "not renting a box that can't download $pin (above)"
    model_bytes=$((model_bytes + bytes))
  done
  read -r auto_disk DOWNLOAD_GB <<<"$(box_disk "$model_bytes" "$BOX_TRAINING")"
  DISK=${DISK:-$auto_disk}
  # One 80 GB GPU, a driver that runs CUDA 13.0, Ampere or Hopper (A100/H100 class: bf16;
  # Blackwell workstation cards, compute capability 12.0, are untested with this vLLM),
  # x86, a verified host with a reliability score above 0.98 (a box that drops mid-run
  # costs a fresh setup), a direct SSH port, >1 Gbps down. No datacenter=True: Vast's
  # datacenter-only tier has no A100s, and the 2026-09-28 run's host was a community one.
  # Use strict '>' and '<': the web UI mishandled '>='.
  QUERY=${FC_VAST_QUERY:-"num_gpus=1 gpu_ram>70 cuda_max_good>12.9 compute_cap>790 compute_cap<1000 cpu_arch=amd64 verified=True reliability>0.98 disk_space>$((DISK - 1)) direct_port_count>0 inet_down>1000"}
  # Appended last. vastai keeps the last clause for a field and operator, and `field=any`
  # drops every earlier clause on that field, so this can tighten a default
  # (reliability>0.99), add one (disk_bw>1000) or drop one (inet_down=any) without
  # restating the query. The price cap is FC_VAST_MAX_DPH, added after this.
  if [ -n "${FC_VAST_QUERY_EXTRA:-}" ]; then
    QUERY="$QUERY $FC_VAST_QUERY_EXTRA"
  fi

  box_plan "$STUDY_LABEL" "$EVAL_REL $MODELS_REL $JUDGE_REL" "$BOX_MODELS" \
    "$(((model_bytes + 500000000) / 1000000000))" "$ADAPTER_REPO"
  echo "   disk: $DISK GB$([ -n "${FC_VAST_DISK:-}" ] && echo " (FC_VAST_DISK)" || echo ", sized from the models (FC_VAST_DISK overrides)")"
  say "cheapest offers under \$$MAX_DPH/hr (GPU plus $DISK GB of storage) for $HOURS GPU hour(s) + $DOWNLOAD_GB GB of downloads, matching: $QUERY"
  offer_search_cmd "$QUERY" "$MAX_DPH" "$DISK"
  offers=$("${SEARCH[@]}" 2>&1 || true)
  picked=$(printf '%s' "$offers" | pick_offer "$HOURS" "$DOWNLOAD_GB" "$MAX_DPH") ||
    die "no offer to rent (nothing is billing)"
  read -r OFFER DPH DOWNLOAD_COST <<<"$picked"
  create=(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct
    --label "$LABEL" --cancel-unavail --raw)
  if [ "$DRY" = 1 ]; then
    say "dry run: every check passed. Would rent offer $OFFER at \$$DPH/hr (setup downloads ~\$$DOWNLOAD_COST) with:"
    echo "   ${create[*]}"
    echo "   then check out ${REF:0:12} from $REPO_URL"
    exit 0
  fi
  confirm "Rent offer $OFFER at \$$DPH/hr, plus ~\$$DOWNLOAD_COST for setup downloads? It bills until scripts/vast/down.sh." || exit 1

  created=$("${create[@]}" 2>&1 || true)
  ID=$(printf '%s' "$created" | json '
if not (isinstance(d, dict) and d.get("new_contract")):
    sys.exit("create failed: %s" % d)
print(d["new_contract"])') ||
    die "Vast returned no instance id (reply above). Check https://cloud.vast.ai/instances/ in case one was created anyway."
  mkdir -p "$STATE_DIR"
  echo "$ID $REF" >"$STATE"
  say "instance $ID created; storage billing starts now, GPU billing once it is running"
  # Account keys only reach boxes created after they were added; attaching covers that.
  attached=$(vastai attach ssh "$ID" "$KEY.pub" 2>&1 || true)
  echo "   attach ssh key: ${attached:0:160}"
fi

# --------------------------------------------------------------------- 6. wait for running
say "waiting for the instance to start (image pull, usually 1-5 min)"
HOST="" PORT="" last_msg="" proxy_polls=0 status=-
for _ in $(seq 90); do
  status=error dph=- host=- port=- kind=- msg=-
  read -r status dph host port kind msg < <(instance_row "$ID" | json '
if not isinstance(d, dict) or d.get("instances", 1) is None:
    print("gone - - - - -")
    sys.exit()
p = ((d.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
ip = (d.get("public_ipaddr") or "").strip()
if p and ip:
    host, port, kind = ip, p, "direct"
elif d.get("ssh_host") and d.get("ssh_port"):
    host, port, kind = d["ssh_host"], d["ssh_port"], "proxy"
else:
    host, port, kind = "-", "-", "-"
print(d.get("actual_status") or "-", "%.3f" % (d.get("dph_total") or 0), host, port, kind,
      " ".join((d.get("status_msg") or "-").split())[:150])' 2>/dev/null || echo "error - - - - -") || true
  if [ "$msg" != - ] && [ "$msg" != "$last_msg" ]; then
    echo "   $status: $msg"
    last_msg=$msg
  fi
  case $status in
    gone) die "instance $ID disappeared (destroyed from the console, or the host dropped it)" ;;
    exited | unknown | offline) die "instance $ID is '$status' and will not start; destroy it with down.sh and try another offer" ;;
  esac
  case $dph in [0-9]*) DPH=$dph ;; esac
  if [ "$status" = running ] && [ "$kind" = direct ]; then
    HOST=$host PORT=$port
    break
  fi
  # Prefer the direct port; fall back to Vast's SSH proxy if it never appears.
  if [ "$status" = running ] && [ "$kind" = proxy ]; then
    proxy_polls=$((proxy_polls + 1))
    if [ "$proxy_polls" -ge 6 ]; then
      warn "no direct SSH port after a minute of running; using Vast's SSH proxy ($host:$port)"
      HOST=$host PORT=$port
      break
    fi
  fi
  sleep 10
done
[ -n "$HOST" ] || die "instance $ID is not running after 15 minutes (last status: $status)"

# A resumed box: what this run will use, and its judge, checked now it is known to run.
if [ -z "$BOX_MODELS" ]; then
  run_files
  judge_preflight
fi

# --------------------------------------------------------------------- 7. ssh vast-em
if ! grep -qx "    HostName $HOST" "$SNIPPET" 2>/dev/null || ! grep -qx "    Port $PORT" "$SNIPPET"; then
  : >"$KNOWN_HOSTS" # a new address: forget the last box's host key (Vast reuses IP:port pairs)
fi
cat >"$SNIPPET" <<EOF
# Written by scripts/vast/up.sh for Vast instance $ID. Rewritten for every new box.
Host vast-em
    HostName $HOST
    Port $PORT
    User root
    IdentityFile "$KEY"
    IdentitiesOnly yes
    ForwardAgent no
    ForwardX11 no
    AddKeysToAgent yes
    IgnoreUnknown UseKeychain
    UseKeychain yes
    UserKnownHostsFile "$KNOWN_HOSTS"
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
chmod 600 "$SNIPPET"
effective=$(ssh -G vast-em 2>/dev/null | awk '$1 == "hostname" {h = $2} $1 == "port" {p = $2} END {print h, p}')
[ "$effective" = "$HOST $PORT" ] ||
  die "ssh vast-em resolves to '$effective', not '$HOST $PORT'. Move 'Include ~/.ssh/vast-em.conf' to the very top of ~/.ssh/config and delete any hand-written 'Host vast-em' block."

say "waiting for sshd at $HOST:$PORT"
for _ in $(seq 60); do
  [ -n "$(ssh-keyscan -T 5 -p "$PORT" "$HOST" 2>/dev/null)" ] && break
  sleep 5
done
say "connecting (your key's passphrase may be asked once)"
connected=0
for _ in 1 2 3 4 5 6; do
  if box true; then
    connected=1
    break
  fi
  sleep 10 # sshd can answer a moment before Vast has installed the keys
done
[ "$connected" = 1 ] ||
  die "ssh vast-em failed. 'Permission denied (publickey)' means the box doesn't have your key: see docs/vast.md troubleshooting."

# Into Hugging Face's own token file, before setup, so setup's downloads, vLLM, Inspect
# and dataset loading all use it. Rewritten on every run, so a resumed box gets it too.
if [ -n "$HF_TOKEN_VALUE" ]; then
  # shellcheck disable=SC2016  # expands on the box
  printf '%s' "$HF_TOKEN_VALUE" |
    box_stdin 'umask 077; d=${HF_HOME:-$HOME/.cache/huggingface}; mkdir -p "$d" && cat >"$d/token" && chmod 600 "$d/token"'
  say "Hugging Face token written to the box's ~/.cache/huggingface/token"
fi

# --------------------------------------------------------------------- 8. box-setup.sh
# done | failed | running | died | none. Last line only, in case a login banner prints first.
setup_state() {
  # shellcheck disable=SC2016  # expands on the box
  box 'if [ -f /root/fc-setup.done ]; then echo done
elif [ -f /root/fc-setup.failed ]; then echo failed
elif [ -f /root/fc-setup.pid ] && kill -0 "$(cat /root/fc-setup.pid)" 2>/dev/null; then echo running
elif [ -f /root/fc-setup.pid ]; then echo died
else echo none; fi' | tail -n 1
}
# Only while setup is not running: bash reads a script as it goes, so overwriting a
# running one can break it.
upload_setup() { box_stdin 'cat >/root/fc-box-setup.sh' <"$REPO/scripts/vast/box-setup.sh"; }
setup_started=0 # 1 once this run starts setup, which then installs what this run asked for
start_setup() {
  setup_started=1
  box 'rm -f /root/fc-setup.done /root/fc-setup.failed /root/fc-setup.pid'
  upload_setup
  box "FC_REPO_URL='$REPO_URL' FC_REF='$REF' FC_BOX_TRAINING='$BOX_TRAINING' FC_BOX_MODELS='$BOX_MODELS' setsid nohup bash -l /root/fc-box-setup.sh >/dev/null 2>&1 </dev/null &"
  say "started box-setup.sh (commit ${REF:0:12})"
}

st=$(setup_state)
case $st in
  done)
    upload_setup # so the check below is this checkout's, not the copy setup ran with
    say "box-setup.sh already finished on this box"
    ;;
  running) say "box-setup.sh is already running on this box" ;;
  failed | died)
    warn "box-setup.sh did not finish on this box before. The end of its log:"
    box 'tail -n 20 /root/fc-setup.log' >&2 || true
    confirm "Upload box-setup.sh from this checkout and run it again?" || exit 1
    start_setup
    st=running
    ;;
  *)
    start_setup
    st=running
    ;;
esac

if [ "$st" != "done" ]; then
  say "waiting for box-setup.sh: uv sync, then ~35 GB of models (usually 10-15 min)."
  echo "   Its log, from another terminal: ssh vast-em tail -f /root/fc-setup.log"
  last_line="" ssh_failures=0
  for _ in $(seq 180); do
    if ! st=$(setup_state 2>/dev/null); then
      ssh_failures=$((ssh_failures + 1))
      [ "$ssh_failures" -lt 6 ] || die "lost SSH to the box (6 tries in a row)"
      sleep 20
      continue
    fi
    ssh_failures=0
    line=$(box 'tail -n 1 /root/fc-setup.log 2>/dev/null' 2>/dev/null || true)
    if [ -n "$line" ] && [ "$line" != "$last_line" ]; then
      echo "   ${line:0:160}"
      last_line=$line
    fi
    case $st in
      done) break ;;
      failed | died)
        warn "box-setup.sh failed. The end of its log:"
        box 'tail -n 30 /root/fc-setup.log' >&2 || true
        die "setup failed. Fix the cause and rerun up.sh (it retries on the same box), or destroy the box with down.sh."
        ;;
    esac
    sleep 20
  done
  [ "$st" = "done" ] || die "box-setup.sh still hasn't finished after an hour"
fi

# --------------------------------------------------------------------- 9. judge key
# Replaced on every run, so a changed judge.env reaches a box that is already running
# (issue #28). box-setup.sh is uploaded again first, so --set-judge exists on the box;
# setup has finished here, so the script isn't running.
upload_setup
box_stdin 'bash /root/fc-box-setup.sh --set-judge' <"$JUDGE_ENV"
say "judge key from $JUDGE_ENV written to the box's .env"

# The adapter-upload token, to its own file (see FC_HF_WRITE_ENV above); without --study,
# any token an earlier run left on a resumed box is removed.
upload_token_to_box "$HF_WRITE_VALUE" "$ADAPTER_REPO"
# Setup runs once per box, so one set up by an earlier run (for another eval, or as an eval
# box now given a study) may lack this run's models or the training envs. Idempotent, so
# cheap when it has them.
if [ "$setup_started" = 0 ]; then
  say "this box was set up by an earlier run; installing and downloading what this run needs that it lacks"
  box "FC_BOX_TRAINING='$BOX_TRAINING' FC_BOX_MODELS='$BOX_MODELS' bash -l /root/fc-box-setup.sh --ensure" ||
    die "could not install or download what this run needs on the box (above)"
fi
# The run's config files, as they are on this laptop, to the same paths in the box's
# checkout: edits made since the commit run on the box without a commit.
for rel in $EVAL_REL $MODELS_REL $JUDGE_REL; do
  box_stdin "cd /root/fragile-compassion && mkdir -p '$(dirname "$rel")' && cat >'$rel'" <"$REPO_P/$rel"
done
say "copied $EVAL_REL, $MODELS_REL and $JUDGE_REL to the box, as they are on this laptop"
[ -z "$STUDY" ] || push_study "$STUDY_DIR"
# What `fc train` records in each run's manifest about the box (no secrets).
python3 -c 'import json, sys; print(json.dumps(dict(zip(
    ["instance_id", "dph_total", "image", "ref", "adapter_repo"], sys.argv[1:]))))' \
  "$ID" "$DPH" "$IMAGE" "$REF" "$ADAPTER_REPO" | box_stdin 'cat >/root/fc-instance.json'

# --------------------------------------------------------------------- 10-11. check, summary
box "FC_BOX_MODELS='$BOX_MODELS' bash -l /root/fc-box-setup.sh --check" || die "the check on the box failed (see above)"
cat <<EOF

== ready: instance $ID at \$$DPH/hr, commit ${REF:0:12}
   Only sshd (port 22) should be listed above as listening publicly.

   ssh vast-em                     # or VS Code: Remote-SSH, host vast-em
   tmux new -s fc                  # after a dropped connection: tmux attach -t fc
   cd fragile-compassion
EOF
if [ -n "$STUDY" ]; then
  echo "   ls $STUDY_LABEL         # the study, as planned on this laptop (docs/training.md)"
fi
# The profile names its models and judge, so only overrides need repeating.
fc_args="--eval $EVAL_REL${MODELS_FILE:+ --models $MODELS_REL}${JUDGE_FILE:+ --judge $JUDGE_REL}"
cat <<EOF
   uv run fc plan $fc_args --run-id <run-id>
   uv run fc run  $fc_args --run-id <run-id>

   When you are done, on this laptop: scripts/vast/down.sh   (the box bills until then)
EOF
