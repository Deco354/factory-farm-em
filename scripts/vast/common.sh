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
