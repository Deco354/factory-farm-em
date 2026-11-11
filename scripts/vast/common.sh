# Shared by up.sh and down.sh; sourced, not run. Paths here are how the two scripts
# find the same box, so they live in one place. Written for bash 3.2 (macOS /bin/bash).
# shellcheck shell=bash
# shellcheck disable=SC2034  # the variables are used by the scripts that source this file

STATE_DIR=$HOME/.config/fragile-compassion
STATE=$STATE_DIR/vast-instance         # "<instance id> <commit>" of the box up.sh rented
SNIPPET=$HOME/.ssh/vast-em.conf        # `Host vast-em`, pulled in by an Include in ~/.ssh/config
KNOWN_HOSTS=$HOME/.ssh/known_hosts_vast
LABEL=fc-em
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
YES=0

# One SSH connection for every call the scripts make, so a key passphrase is asked at
# most once per run, even without an agent. Only the scripts use it, not `ssh vast-em`.
CM=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-vast-em-%C" -o ControlPersist=15m
  -o ConnectTimeout=15)

# What down.sh treats as work still running on the box (`pgrep -f`): evals, vLLM and
# training. One letter of each is in brackets so the pattern can't match pgrep's own
# command line, which contains the pattern. TRAINING_PATTERN is the part that writes a
# study's manifests (training, uploads, selftests), so up.sh copies no study meanwhile.
TRAINING_PATTERN='[f]c train|[a]xolotl'
BUSY_PATTERN="[f]c run|[i]nspect eval|[v]llm serve|$TRAINING_PATTERN"

say() { echo "== $*"; }
warn() { echo "!! $*" >&2; }
die() {
  echo "error: $*" >&2
  exit 1
}

# laptop_only: up.sh and down.sh drive a box from your laptop, and stop at once on a box
# box-setup.sh set up, where the repo is checked out too. Run there, down.sh found no
# record of a rented box in the box's own home and listed the box itself as billing, to be
# destroyed by hand before its results were copied (issue #100). The box is told by
# box-setup.sh's log, which is in its home (/root, as SSH logs in) and on no laptop.
laptop_only() {
  local script
  script=$(basename "$0")
  [ ! -e "$HOME/fc-setup.log" ] ||
    die "$script runs on your laptop, not on the box: this machine has $HOME/fc-setup.log, box-setup.sh's log, so it is a box up.sh set up. Type exit to leave the SSH session, then run scripts/vast/$script from the repo on your laptop."
}

# size_kb KB: a size in KB (1024 bytes, as `du -k` counts) for a person to read.
size_kb() {
  awk -v kb="$1" 'BEGIN {
    b = kb * 1024
    if (b >= 1e9) printf "%.1f GB\n", b / 1e9
    else if (b >= 1e6) printf "%.0f MB\n", b / 1e6
    else printf "%.0f KB\n", b / 1e3
  }'
}

# start_progress DIR TOTAL_KB: until stop_progress, every PROGRESS_EVERY seconds, prints
# how much has been copied into DIR so far, of about TOTAL_KB, so a long copy doesn't look
# like a hang. It counts the files changed since it started, by change time: tar sets each
# file's modification time back to the box's, and a file an earlier copy left there, now
# overwritten, counts again. In the background; the EXIT trap stops it too.
PROGRESS_EVERY=15 PROGRESS_PID="" PROGRESS_MARK=""
start_progress() {
  PROGRESS_MARK=$(mktemp "${TMPDIR:-/tmp}/fc-copy-start.XXXXXX")
  (
    nap=""
    trap 'kill "$nap" 2>/dev/null || true; exit 0' TERM
    while :; do
      # A background sleep and `wait`, so TERM stops it at once. Its output goes nowhere,
      # so a sleep left behind can't hold the script's output open.
      sleep "$PROGRESS_EVERY" >/dev/null 2>&1 &
      nap=$!
      wait "$nap" || true
      kb=$(find "$1" -type f -cnewer "$PROGRESS_MARK" -exec du -k {} + 2>/dev/null |
        awk '{kb += $1} END {print kb + 0}') || true
      echo "   $(size_kb "$kb") of about $(size_kb "$2") copied so far"
    done
  ) &
  PROGRESS_PID=$!
}
stop_progress() {
  if [ -n "$PROGRESS_PID" ]; then
    kill "$PROGRESS_PID" 2>/dev/null || true
    wait "$PROGRESS_PID" 2>/dev/null || true
  fi
  [ -z "$PROGRESS_MARK" ] || rm -f "$PROGRESS_MARK"
  PROGRESS_PID="" PROGRESS_MARK=""
}

# billing_traps: from now on, a failure, Ctrl-C or kill once a box is rented ($ID set)
# says the box is still billing. INT and TERM need traps of their own: on Ctrl-C bash
# runs the EXIT trap with $? from the last finished command, usually 0, so the warning
# was skipped (macOS bash 3.2, 2026-10-06). Exiting 130/143 makes the status say so.
# The EXIT trap also stops a progress ticker: bash makes background commands ignore
# Ctrl-C, so it would otherwise outlive the script.
billing_traps() {
  trap billing_warning EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}
billing_warning() {
  local rc=$?
  stop_progress
  if [ "$rc" -ne 0 ] && [ -n "${ID:-}" ]; then
    still_billing_banner "$ID" "${DPH:-}"
  fi
}

# banner LINE...: the lines framed by !! rules on stderr, for messages that must not be
# missed, even straight after a ^C. Red and bold on a terminal unless NO_COLOR is set;
# plain text otherwise. An empty LINE is a spacer.
banner() {
  local on="" off="" rule line
  if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
    on=$'\033[1;31m' off=$'\033[0m'
  fi
  rule='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  {
    echo
    echo "$on$rule"
    for line in "$@"; do
      if [ -n "$line" ]; then echo "!!  $line"; else echo "!!"; fi
    done
    echo "$rule$off"
    echo
  } >&2
}

# still_billing_banner ID [DPH]
still_billing_banner() {
  local price=""
  case ${2:-} in [0-9]*) price=" at \$$2/hr" ;; esac
  banner "THE VAST BOX IS STILL RUNNING AND BILLING: instance $1$price" "" \
    "Pick it up again:  scripts/vast/up.sh" \
    "Destroy it:        scripts/vast/down.sh" \
    "See it:            https://cloud.vast.ai/instances/"
}

# confirm "question" -> 0 on y/Y. --yes answers for you; no terminal counts as "no".
confirm() {
  local answer
  [ "$YES" = 1 ] && return 0
  read -r -p "$1 [y/N] " answer || return 1
  [ "$answer" = y ] || [ "$answer" = Y ]
}

# Callers pass one command string, expanded here on purpose (the values are ours).
# shellcheck disable=SC2029
box() { ssh -n "${CM[@]}" vast-em "$@"; } # run on the box, stdin closed
# shellcheck disable=SC2029
box_stdin() { ssh "${CM[@]}" vast-em "$@"; } # run on the box, reading the caller's stdin

# json 'python code' [args] < json: runs the code with the parsed input as `d` and the
# args as sys.argv[2:]. vastai exits 0
# on most failures and reports them as JSON on stderr, so callers capture 2>&1 and
# check the content, never the exit code.
json() {
  python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("expected JSON from vastai, got: " + (raw.strip()[:300] or "nothing"))
exec(sys.argv[1])
' "$@"
}

# SSH_KEY_FIX: the start-up command up.sh gives every box (`create instance --onstart-cmd`).
# Some Vast hosts write /root/.ssh/authorized_keys owned by a user the container doesn't
# have (`UNKNOWN:UNKNOWN 644`), and sshd refuses it ("Authentication refused: bad ownership
# or modes for file /root/.ssh/authorized_keys" in `vastai logs`), so no key works. Seen on
# 2026-10-30 on host 203531 (twice, and fixed by this on a third box) and one other; a
# Czech host wrote it correctly. For the box's first ten minutes, so that keys Vast writes
# after start-up are covered too, it makes /root, /root/.ssh and the file owned by root and
# writable only by root. The first state it finds goes to /root/fc-ssh-modes.log. It runs as
# root on the box, under sh; its first line holds its settings (tests change them).
# shellcheck disable=SC2016  # expands on the box
SSH_KEY_FIX='R=/root O=root:root N=120 S=5
(logged=0
for i in $(seq "$N"); do
  if [ -f "$R/.ssh/authorized_keys" ]; then
    if [ "$logged" = 0 ]; then
      stat -c "%U:%G %a %n" "$R" "$R/.ssh" "$R/.ssh/authorized_keys" >"$R/fc-ssh-modes.log" 2>&1
      logged=1
    fi
    chown "$O" "$R" "$R/.ssh" "$R/.ssh/authorized_keys"
    chmod go-w "$R"
    chmod 700 "$R/.ssh"
    chmod 600 "$R/.ssh/authorized_keys"
  fi
  sleep "$S"
done) >/dev/null 2>&1 &'

# offer_search_cmd QUERY MAX_DPH DISK: sets SEARCH to the offer search up.sh runs.
# --storage prices every offer with the disk that will be rented. vastai's default is
# 5 GB, which left dph_total, the price cap and the ranking about $0.05/hr low on a
# 120 GB box, and different from the web console, which prices its own disk filter.
# Kept here, not inline in up.sh, so tests can check the command without running up.sh.
offer_search_cmd() {
  SEARCH=(vastai search offers "$1 dph_total<$2" --storage "$3" -o dph --limit 50 --raw)
}

# pick_offer HOURS DOWNLOAD_GB MAX_DPH < the offer search's reply: prints the cheapest
# offer as "id dph_total download_cost" and lists the three cheapest on stderr. Ranked by
# dph_total x HOURS + inet_down_cost x DOWNLOAD_GB. The price cap is checked again on what
# Vast returns. Fails, saying why, when the search failed or nothing is under the cap.
# Kept here, not inline in up.sh, so tests can check the choice on hand-written offers.
pick_offer() {
  json '
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
# machines this search never returns (docs/vast.md troubleshooting).
for o in d[:3]:
    print("   offer %-10s %-18s $%.3f/hr  downloads $%5.2f/TB  ~$%.2f  %-20s m:%s host:%s  reliability %.3f  down %s Mb/s" % (
        o["id"], o["gpu_name"], o["dph_total"], (o.get("inet_down_cost") or 0) * 1000, estimate(o),
        (o.get("geolocation") or "?")[:20], o.get("machine_id", "?"), o.get("host_id", "?"),
        o.get("reliability") or 0, round(o.get("inet_down") or 0)),
        file=sys.stderr)
best = d[0]
print(best["id"], "%.3f" % best["dph_total"], "%.2f" % (gb * (best.get("inet_down_cost") or 0)))' "$@"
}

# pin_bytes REPO@COMMIT < Hugging Face's reply to
# GET https://huggingface.co/api/models/REPO/revision/COMMIT?blobs=true: prints how many
# bytes box-setup.sh downloads for it (every file at that commit: snapshot_download takes
# whole repos), or says why not and returns 1. A repo or commit Hugging Face doesn't
# know, or one it won't show without a token, fails here before renting, not on the box.
pin_bytes() {
  python3 -c '
import json, sys
pin = sys.argv[1]
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("%s: expected JSON from Hugging Face, got: %s" % (pin, raw.strip()[:200] or "nothing"))
if not isinstance(d, dict) or "error" in d:
    why = d.get("error") if isinstance(d, dict) else raw.strip()[:200]
    sys.exit("%s: Hugging Face has no such repo and commit, or shows it only to a token that "
             "may read it (hf.env): %s" % (pin, why))
if d.get("sha") != pin.rsplit("@", 1)[1]:
    sys.exit("%s: Hugging Face answered for commit %s" % (pin, d.get("sha")))
files = d.get("siblings") or []
if not files or not all(isinstance(f.get("size"), int) for f in files):
    sys.exit("%s: Hugging Face listed no file sizes" % pin)
print(sum(f["size"] for f in files))' "$1"
}

# box_disk MODEL_BYTES TRAINING: prints "<disk GB> <download GB>" for a box that
# pre-downloads MODEL_BYTES of models, with the training envs when TRAINING is 1.
# Downloads are the models plus ~10 GB of wheels (~10 GB more for the training envs). The
# disk holds the models plus 50 GB (70 GB with the training envs) for the venv, the image,
# logs and checkpoints, and never less than 60 GB. Estimates: the only measured box
# (hb-em-smoke-002, docs/gpu-runs.md) used 106 GB for ~108 GB of models and the venv;
# `df -h /` on the next one says more. FC_VAST_DISK overrides the disk.
box_disk() {
  local models_gb=$((($1 + 999999999) / 1000000000)) wheels=10 room=50 disk
  if [ "$2" = 1 ]; then
    wheels=20 room=70
  fi
  disk=$((models_gb + room))
  [ "$disk" -ge 60 ] || disk=60
  echo "$disk $((models_gb + wheels))"
}

# new_pins RECORD PINS: prints the space-separated pins in PINS that RECORD (the box's
# /root/fc-box-models: every model its setup and --ensure runs were given) lacks. Returns 1
# if RECORD can't tell: empty, or holding models.yaml paths, as boxes set up before up.sh
# sent pins recorded.
new_pins() {
  local record=" ${1//[[:space:]]/ } " item out=""
  [ -n "${1//[[:space:]]/}" ] || {
    echo "can't tell which models this box already has: its record is empty" >&2
    return 1
  }
  for item in $record; do
    case $item in
      *@*) ;;
      *)
        echo "can't tell which models this box already has: its record names $item" >&2
        return 1
        ;;
    esac
  done
  for item in $2; do
    case $record in *" $item "*) ;; *) out="${out:+$out }$item" ;; esac
  done
  echo "$out"
}

# disk_problem NEED_BYTES FREE_KB: says why NEED_BYTES of downloads don't fit in a resumed
# box's FREE_KB (`df -Pk`) and returns 1, or returns 0. The box was sized for the models it
# was rented with, so a run adding more could fill it (issue #81). 10 GB must stay free for
# the run itself (logs, vLLM's caches, checkpoints): a guess, as box_disk's room is.
disk_problem() {
  local need_gb=$((($1 + 999999999) / 1000000000)) free_gb=$(($2 * 1024 / 1000000000)) room=10
  [ $((need_gb + room)) -gt "$free_gb" ] || return 0
  echo "this box has $free_gb GB free, and this run adds about $need_gb GB of downloads, plus $room GB to leave free for the run" >&2
  return 1
}

# hf_token_problem < whoami-v2 reply: prints why a Hugging Face token must not go on a
# rented box and returns 1, or returns 0 for a read-only one. Classic tokens have role
# `read` or `write`. Fine-grained ones list permissions (globally and per scope), and
# every one must end in `.read`, which rules out writing and paid inference calls.
# Reading gated repos is a separate flag (`canReadGatedRepos`), not a permission.
hf_token_problem() {
  python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("expected JSON from Hugging Face, got: " + (raw.strip()[:200] or "nothing"))
if "error" in d:
    sys.exit("Hugging Face rejected the token: %s" % d["error"])
token = (d.get("auth") or {}).get("accessToken") or {}
role = token.get("role")
if role == "read":
    sys.exit(0)
if role == "fineGrained":
    fine = token.get("fineGrained") or {}
    perms = list(fine.get("global") or [])
    for scope in fine.get("scoped") or []:
        perms += scope.get("permissions") or []
    beyond = sorted({p for p in perms if not p.endswith(".read")})
    if beyond:
        sys.exit("the token can do more than read: %s" % ", ".join(beyond))
    sys.exit(0)
sys.exit("the token is not read-only (role: %s)" % role)'
}

# hf_write_token_problem REPO < whoami-v2 reply: prints why a Hugging Face token must not
# go on a rented box as the adapter-upload token and returns 1, or returns 0. Only a
# fine-grained token passes, and its only permission that doesn't end in `.read` must be
# `repo.write`, scoped to REPO alone: a host that reads the token can then write to that
# one private repo and nothing else. Fails closed: a permission or scope this doesn't
# recognise is refused and printed. (`repo.write` and the scope's `type: model` are the
# names the plan expects; the first real token checked confirms them.)
hf_write_token_problem() {
  python3 -c '
import json, sys
repo = sys.argv[1]
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("expected JSON from Hugging Face, got: " + (raw.strip()[:200] or "nothing"))
if "error" in d:
    sys.exit("Hugging Face rejected the token: %s" % d["error"])
token = (d.get("auth") or {}).get("accessToken") or {}
if token.get("role") != "fineGrained":
    sys.exit("the upload token must be fine-grained, scoped to %s (role: %s)" % (repo, token.get("role")))
fine = token.get("fineGrained") or {}
beyond = ["global: %s" % p for p in fine.get("global") or [] if not p.endswith(".read")]
can_write_repo = False
for scope in fine.get("scoped") or []:
    entity = scope.get("entity") or {}
    perms = scope.get("permissions") or []
    is_repo = entity.get("type") == "model" and entity.get("name") == repo
    for p in perms:
        if p.endswith(".read"):
            continue
        if p == "repo.write" and is_repo:
            can_write_repo = True
            continue
        beyond.append("%s %s: %s" % (entity.get("type"), entity.get("name"), p))
if beyond:
    sys.exit("the token can do more than write to %s: %s" % (repo, "; ".join(sorted(beyond))))
if not can_write_repo:
    sys.exit("the token has no repo.write permission scoped to %s" % repo)' "$1"
}

# study_archive RUNS_DIR REL: RUNS_DIR/REL as a tar on stdout, for the box, without its
# _candidates/ (plan's work dir). COPYFILE_DISABLE and --no-xattrs keep macOS metadata out:
# without them the box's GNU tar warned "Ignoring unknown extended header keyword
# 'LIBARCHIVE.xattr.com.apple.provenance'" once per file (seen 2026-10-30).
study_archive() {
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$1" --exclude="$2/_candidates" -cf - "$2"
}

# study_settings DIR: what a training box needs for the study `fc train plan` wrote to DIR,
# read from its runs' manifests: prints "<base model>@<revision> <adapter repo> <commit>".
# Fails, saying why, when DIR holds no planned runs, when the runs disagree (the parity
# gate makes them agree, so that means a damaged or hand-edited plan), when a run was
# planned before manifests recorded the adapter repo, or when the study was planned with
# uncommitted changes: the box checks out the commit without them, so it would train
# other code than the plan recorded.
study_settings() {
  python3 -c '
import glob, json, os, sys
study = sys.argv[1]
paths = sorted(glob.glob(os.path.join(study, "*", "manifest.json")))
if not paths:
    sys.exit("%s holds no planned runs (no */manifest.json); plan it with fc train plan" % study)
seen = {}
for path in paths:
    run = os.path.basename(os.path.dirname(path))
    try:
        with open(path) as f:
            m = json.load(f)
        values = (m["model"]["base_model"], m["model"]["base_revision"],
                  m["run"].get("adapter_repo"), m["code"]["git_commit"])
        dirty = m["code"]["dirty"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        sys.exit("%s: not a manifest fc train plan wrote (%s: %s)" % (path, type(exc).__name__, exc))
    if values[2] is None:
        sys.exit("run %s was planned before manifests recorded the adapter repo; plan the study "
                 "again with fc train plan --replace" % run)
    if dirty:
        sys.exit("run %s was planned with uncommitted changes, which the box would not have; "
                 "commit them, push, and plan the study again" % run)
    seen.setdefault(values, []).append(run)
if len(seen) > 1:
    sys.exit("the runs disagree on base model, revision, adapter repo or commit: " + "; ".join(
        "%s: %s@%s %s %s" % ((", ".join(runs),) + values) for values, runs in seen.items()))
base, revision, repo, commit = next(iter(seen))
print("%s@%s %s %s" % (base, revision, repo, commit))' "$1"
}

# adapter_repo_problem REPO < the reply to GET https://huggingface.co/api/models/REPO, made
# with the upload token: prints why adapters must not go to REPO and returns 1, or returns 0
# when it is a private model repo the token can see. Hugging Face answers a private repo
# the token can't see as if it didn't exist.
adapter_repo_problem() {
  python3 -c '
import json, sys
repo = sys.argv[1]
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("expected JSON from Hugging Face, got: " + (raw.strip()[:200] or "nothing"))
if not isinstance(d, dict):
    sys.exit("expected a repo from Hugging Face, got: " + raw.strip()[:200])
if "error" in d:
    sys.exit("the upload token sees no model repo %s (Hugging Face: %s). Create it as a "
             "private model repo, and check the token is scoped to it" % (repo, d["error"]))
if d.get("private") is not True:
    sys.exit("%s is public; adapters go only to a private repo. Make it private first" % repo)' "$1"
}

# read_repo_problem REPO < the same GET, made with the box's read-only token (hf.env): prints
# why the box's evals couldn't load adapters back from REPO and returns 1, or returns 0 when
# the token can see it. A --study box's evals download the uploads with that token.
read_repo_problem() {
  python3 -c '
import json, sys
repo = sys.argv[1]
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    sys.exit("expected JSON from Hugging Face, got: " + (raw.strip()[:200] or "nothing"))
if not isinstance(d, dict):
    sys.exit("expected a repo from Hugging Face, got: " + raw.strip()[:200])
if "error" in d:
    sys.exit("the read token sees no model repo %s (Hugging Face: %s), so the evals could "
             "not load the adapters uploaded there. Give it read access to the repo" % (repo, d["error"]))' "$1"
}

# box_plan STUDY CONFIGS PINS MODELS_GB ADAPTER_REPO: what a new box will be set up for,
# printed before renting, dry runs included. STUDY is the --study directory, empty for an
# evaluation-only box; CONFIGS the config files copied to it; PINS the repo@commit pins it
# pre-downloads, MODELS_GB in all; ADAPTER_REPO where the study's adapters upload.
box_plan() {
  local pin
  if [ -n "$1" ]; then
    say "box: training and evaluation, for $1 (setup also installs the training envs)"
  else
    say "box: evaluation only (no training envs; up.sh --study runs/<study> rents a training box)"
  fi
  echo "   configs, copied to the box as they are here: $2"
  echo "   pre-downloads, $4 GB:"
  for pin in $3; do
    echo "     ${pin%@*} @ $(printf '%.12s' "${pin##*@}")"
  done
  if [ -n "$5" ]; then
    echo "   adapter uploads: to $5"
  else
    echo "   adapter uploads: none (the upload token goes only to a --study box)"
  fi
}

# upload_token_to_box TOKEN REPO: the adapter-upload token to its own file on the box
# (mode 600). With TOKEN empty (not a --study run) it removes any token an earlier run left
# there, so a box holds one only while it is being set up for a study.
upload_token_to_box() {
  local gone
  if [ -n "$1" ]; then
    printf '%s' "$1" |
      box_stdin 'umask 077; mkdir -p /root/.config/fc && cat >/root/.config/fc/hf-write-token && chmod 600 /root/.config/fc/hf-write-token'
    say "upload token for $2 written to the box's /root/.config/fc/hf-write-token"
  else
    # shellcheck disable=SC2016  # expands on the box
    gone=$(box 'f=/root/.config/fc/hf-write-token; if [ -f "$f" ]; then rm "$f" && echo removed; fi') ||
      die "could not remove the upload token an earlier run left in the box's /root/.config/fc/hf-write-token"
    [ "$gone" != removed ] || say "removed the upload token an earlier run left on the box (only a --study run sends one)"
  fi
}

# The instance's JSON row; `{"instances": null}` once it no longer exists.
instance_row() { vastai show instance "$1" --raw 2>&1 || true; }

# 0 if the row says the instance is gone: `{"instances": null}`, which is vastai's reply
# for an instance that no longer exists (vastai 1.8.2 show__instance). Every error counts
# as not gone, a 404 included: vastai also answers 404 for an expired 2FA session, and a
# box wrongly taken as gone would keep billing with no record of it.
instance_gone() {
  printf '%s' "$1" | json '
gone = isinstance(d, dict) and "instances" in d and d["instances"] is None
sys.exit(0 if gone else 1)' 2>/dev/null
}
