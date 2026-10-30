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
# command line, which contains the pattern.
BUSY_PATTERN='[f]c run|[f]c train|[i]nspect eval|[v]llm serve|[a]xolotl'

say() { echo "== $*"; }
warn() { echo "!! $*" >&2; }
die() {
  echo "error: $*" >&2
  exit 1
}

# billing_traps: from now on, a failure, Ctrl-C or kill once a box is rented ($ID set)
# says the box is still billing. INT and TERM need traps of their own: on Ctrl-C bash
# runs the EXIT trap with $? from the last finished command, usually 0, so the warning
# was skipped (macOS bash 3.2, 2026-10-06). Exiting 130/143 makes the status say so.
billing_traps() {
  trap billing_warning EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}
billing_warning() {
  local rc=$?
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
