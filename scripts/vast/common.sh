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

# still_billing_banner ID [DPH]: a warning on stderr that is hard to miss, even straight
# after a ^C. Red and bold on a terminal unless NO_COLOR is set; plain text otherwise.
still_billing_banner() {
  local id=$1 price="" on="" off="" rule
  case ${2:-} in [0-9]*) price=" at \$$2/hr" ;; esac
  if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
    on=$'\033[1;31m' off=$'\033[0m'
  fi
  rule='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  {
    echo
    echo "$on$rule"
    echo "!!  THE VAST BOX IS STILL RUNNING AND BILLING: instance $id$price"
    echo "!!"
    echo "!!  Pick it up again:  scripts/vast/up.sh"
    echo "!!  Destroy it:        scripts/vast/down.sh"
    echo "!!  See it:            https://cloud.vast.ai/instances/"
    echo "$rule$off"
    echo
  } >&2
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

# The instance's JSON row; `{"instances": null}` once it no longer exists.
instance_row() { vastai show instance "$1" --raw 2>&1 || true; }

# 0 if the row says the instance is gone: `{"instances": null}`, or a 404 error.
instance_gone() {
  printf '%s' "$1" | json '
gone = isinstance(d, dict) and (
    ("instances" in d and d["instances"] is None) or d.get("status_code") == 404)
sys.exit(0 if gone else 1)' 2>/dev/null
}
