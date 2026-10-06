#!/usr/bin/env bash
# Rent a Vast.ai GPU box and get it ready for `fc run`. Runs on your laptop (macOS or Linux).
#
#   scripts/vast/up.sh [--dry-run] [--yes]
#
# First checks everything it can for free: tools, SSH config, that the commit is on
# GitHub, and one real judge call made with only the box's key. Then it shows the
# cheapest matching offer, counting GPU time and setup downloads, rents it once you
# confirm, and points `ssh vast-em` at it.
# It then runs box-setup.sh there (uv sync, all pinned models) and copies the judge key
# over. If it fails or you press Ctrl-C after renting, run it again: it picks up the
# recorded box instead of renting a second one. See README "Rented GPU box (Vast.ai)".
#
#   --dry-run   run every check and the offer search, then stop without renting
#   --yes       answer yes to every question (for scripted use)
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
#   FC_VAST_DISK    disk in GB                        (120)
#   FC_VAST_QUERY   `vastai search offers` filter     (see QUERY below; replaces it all)
#   FC_VAST_QUERY_EXTRA clauses added after the query (e.g. 'reliability>0.99 inet_down=any')
#   FC_VAST_HOURS   GPU hours used to rank offers     (1; raise it for long runs)
#   FC_VAST_MAX_DPH most $/hr it will ever rent at    (3.00 for GPU + disk; applies even with --yes)
set -euo pipefail

# shellcheck source=scripts/vast/common.sh
. "$(dirname "$0")/common.sh"

# --------------------------------------------------------------------- settings file
# One FC_NAME=value per line; '#' starts a comment line; one pair of surrounding quotes
# is removed; nothing is expanded or executed. Unknown names fail, so a typo can't be
# silently ignored.
SETTINGS=${FC_VAST_SETTINGS:-$STATE_DIR/vast.env}
KNOWN_SETTINGS=" FC_REF FC_REPO_URL FC_JUDGE_ENV FC_HF_ENV FC_VAST_SSH_KEY FC_VAST_IMAGE FC_VAST_DISK FC_VAST_QUERY FC_VAST_QUERY_EXTRA FC_VAST_HOURS FC_VAST_MAX_DPH "
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

JUDGE_ENV=${FC_JUDGE_ENV:-$STATE_DIR/judge.env}
# Optional: a fine-grained, read-only Hugging Face token for the box (README one-time
# setup step 7). It authenticates the box's downloads, so fewer rate limits, and gated
# datasets such as TAC's need it. Read only from this file, never your shell's HF_TOKEN;
# checked before renting; sent to the box over SSH stdin, never on a command line.
HF_ENV=${FC_HF_ENV:-$STATE_DIR/hf.env}
HF_TOKEN_VALUE=""
if [ -f "$HF_ENV" ]; then
  [ "$(grep -c '^HF_TOKEN=.' "$HF_ENV")" = 1 ] || die "$HF_ENV must hold exactly one HF_TOKEN=<token> line"
  HF_TOKEN_VALUE=$(sed -n 's/^HF_TOKEN=//p' "$HF_ENV")
  HF_TOKEN_VALUE=${HF_TOKEN_VALUE%$'\r'}
  case $HF_TOKEN_VALUE in
    \"*\") HF_TOKEN_VALUE=${HF_TOKEN_VALUE#\"} HF_TOKEN_VALUE=${HF_TOKEN_VALUE%\"} ;;
    \'*\') HF_TOKEN_VALUE=${HF_TOKEN_VALUE#\'} HF_TOKEN_VALUE=${HF_TOKEN_VALUE%\'} ;;
  esac
  chmod 600 "$HF_ENV"
fi
KEY=${FC_VAST_SSH_KEY:-$HOME/.ssh/vastai}
# Pinned by date: the matching `cuda-13.0.3-auto` tag moves. CUDA 13.0 matches the
# lockfile's torch/vLLM runtime. SSH launch mode skips the image's own entrypoint, so
# none of its web services (Instance Portal, Jupyter, Syncthing) start.
IMAGE=${FC_VAST_IMAGE:-vastai/base-image:cuda-13.0.3-cudnn-devel-ubuntu24.04-2026-09-07}
DISK=${FC_VAST_DISK:-120}
case $DISK in '' | *[!0-9]*) die "FC_VAST_DISK must be a whole number of GB, got '$DISK'" ;; esac
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
# Offers are ranked by GPU price x FC_VAST_HOURS + download price x DOWNLOAD_GB: hosts
# charge 0-40 $/TB for downloads, which the hourly price leaves out, and a new box
# downloads about 45 GB (models, wheels) before it is ready.
HOURS=${FC_VAST_HOURS:-1}
DOWNLOAD_GB=45
# A cap on the hourly price, outside FC_VAST_QUERY so overriding the query keeps it. It
# is sent to Vast and checked again on what comes back.
MAX_DPH=${FC_VAST_MAX_DPH:-3.00}
for setting in "FC_VAST_HOURS=$HOURS" "FC_VAST_MAX_DPH=$MAX_DPH"; do
  case ${setting#*=} in
    '' | .* | *. | *[!0-9.]* | *.*.*) die "${setting%%=*} must be a number, got '${setting#*=}'" ;;
  esac
done

DRY=0
for arg in "$@"; do
  case $arg in
    --dry-run) DRY=1 ;;
    --yes | -y) YES=1 ;;
    -h | --help)
      awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0" # the header comment

      exit 0
      ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done

ID=""
DPH="?"
billing_traps # from here on, any exit with a box rented says it is still billing

# --------------------------------------------------------------------- 1. local checks
for tool in vastai python3 ssh ssh-keyscan scp git uv curl; do
  command -v "$tool" >/dev/null || die "$tool not found on PATH (README: Rented GPU box, one-time setup)"
done
[ -f "$KEY" ] && [ -f "$KEY.pub" ] || die "no SSH key pair at $KEY and $KEY.pub (one-time setup step 4, or set FC_VAST_SSH_KEY)"
[ -f "$JUDGE_ENV" ] || die "no judge key file at $JUDGE_ENV (one-time setup step 6)"
grep -Eq '^[A-Za-z_][A-Za-z0-9_]*=.' "$JUDGE_ENV" || die "$JUDGE_ENV has no NAME=value line"
chmod 600 "$JUDGE_ENV"
grep -Eiq '^[[:space:]]*include[[:space:]].*vast-em\.conf' "$HOME/.ssh/config" 2>/dev/null ||
  die "add this as the first line of ~/.ssh/config:  Include ~/.ssh/vast-em.conf"

cd "$REPO"
if [ -f "$STATE" ] && [ "$DRY" = 0 ]; then
  # ----------------------------------------------------------------- resume a recorded box
  read -r ID REF <"$STATE" || true
  [ -n "$ID" ] && [ -n "${REF:-}" ] || die "$STATE is malformed; check https://cloud.vast.ai/instances/ and delete it"
  say "resuming instance $ID, commit ${REF:0:12} (recorded in $STATE)"
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
  if [ -z "${FC_REF:-}" ] && [ -n "$(git status --porcelain)" ]; then
    warn "you have uncommitted changes; the box gets commit ${REF:0:12} without them:"
    git status --short | head -n 10 >&2
    confirm "Continue anyway?" || exit 1
  fi
  REPO_URL=${FC_REPO_URL:-$(git remote get-url origin)}

  # ----------------------------------------------------------------- 3. judge preflight
  # One real call to the judge in configs/judge.yaml, in an environment holding nothing
  # but $JUDGE_ENV (Inspect's get_model() does not read .env files). A wrong, revoked or
  # unfunded key fails here instead of stalling `fc run` on a rented GPU.
  say "judge check: one call to the configs/judge.yaml judge, using only $JUDGE_ENV"
  tmp=$(mktemp -d)
  judge_ok=1
  (cd "$tmp" && env -i HOME="$HOME" PATH="$PATH" TERM="${TERM:-dumb}" \
    uv run --quiet --project "$REPO" --env-file "$JUDGE_ENV" \
    python - "$REPO/configs/judge.yaml" <<'PY'
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
  [ "$judge_ok" = 1 ] || die "judge check failed; fix $JUDGE_ENV before renting (README troubleshooting)"
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
    die "not putting the token in $HF_ENV on a rented box (above); see README one-time setup step 7"
  echo "   read-only: OK"
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

  # ----------------------------------------------------------------- 5. offer, then rent
  say "cheapest offers under \$$MAX_DPH/hr (GPU plus $DISK GB of storage) for $HOURS GPU hour(s) + $DOWNLOAD_GB GB of downloads, matching: $QUERY"
  offer_search_cmd "$QUERY" "$MAX_DPH" "$DISK"
  offers=$("${SEARCH[@]}" 2>&1 || true)
  picked=$(printf '%s' "$offers" | json '
hours, gb, max_dph = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if not isinstance(d, list):
    sys.exit("offer search failed: %s" % d)
d = [o for o in d if o["dph_total"] < max_dph]
if not d:
    sys.exit("no offers match under $%.2f/hr; try again later, or raise FC_VAST_MAX_DPH or relax FC_VAST_QUERY" % max_dph)
def estimate(o):
    return o["dph_total"] * hours + gb * (o.get("inet_down_cost") or 0)
d.sort(key=estimate)
# m: and host: are the numbers the web console shows. The offer id is one free GPU slot
# on the machine, so the console can show another id for the same machine. The search
# API also bundles similar offers and returns one per bundle, so the console can list
# machines this search never returns (README troubleshooting).
for o in d[:3]:
    print("   offer %-10s %-18s $%.3f/hr  downloads $%5.2f/TB  ~$%.2f  %-20s m:%s host:%s  reliability %.3f  down %s Mb/s" % (
        o["id"], o["gpu_name"], o["dph_total"], (o.get("inet_down_cost") or 0) * 1000, estimate(o),
        (o.get("geolocation") or "?")[:20], o.get("machine_id", "?"), o.get("host_id", "?"),
        o.get("reliability") or 0, round(o.get("inet_down") or 0)),
        file=sys.stderr)
best = d[0]
print(best["id"], "%.3f" % best["dph_total"], "%.2f" % (gb * (best.get("inet_down_cost") or 0)))' \
    "$HOURS" "$DOWNLOAD_GB" "$MAX_DPH") ||
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
  die "ssh vast-em failed. 'Permission denied (publickey)' means the box doesn't have your key: see README troubleshooting."

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
start_setup() {
  box 'rm -f /root/fc-setup.done /root/fc-setup.failed /root/fc-setup.pid'
  upload_setup
  box "FC_REPO_URL='$REPO_URL' FC_REF='$REF' setsid nohup bash -l /root/fc-box-setup.sh >/dev/null 2>&1 </dev/null &"
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
marker="# judge credential (appended by scripts/vast/up.sh)"
box_stdin "umask 077; cd /root/fragile-compassion && if ! grep -qxF '$marker' .env; then { echo; echo '$marker'; cat; } >>.env; fi" <"$JUDGE_ENV"
say "judge key is in the box's .env"

# --------------------------------------------------------------------- 10-11. check, summary
box 'bash -l /root/fc-box-setup.sh --check' || die "the check on the box failed (see above)"
cat <<EOF

== ready: instance $ID at \$$DPH/hr, commit ${REF:0:12}
   Only sshd (port 22) should be listed above as listening publicly.

   ssh vast-em                     # or VS Code: Remote-SSH, host vast-em
   tmux new -s fc                  # after a dropped connection: tmux attach -t fc
   cd fragile-compassion
   uv run fc plan --eval configs/eval.smoke.yaml --run-id <run-id>
   uv run fc run  --eval configs/eval.smoke.yaml --run-id <run-id>

   When you are done, on this laptop: scripts/vast/down.sh   (the box bills until then)
EOF
