#!/usr/bin/env bash
# Copy results off the rented Vast.ai box, then destroy it. Runs on your laptop.
#
#   scripts/vast/down.sh [--keep] [--yes] [dest-dir]   (default: logs/vast-<date>-<instance>/)
#
# Copies the box's logs/, results/ and runs/ (training bundles, without optimizer states
# or prepared datasets), plus the setup log and a patch of any edits made on the box,
# into dest-dir, destroys the instance without asking, and waits until
# Vast confirms it is gone. It says so as soon as it starts, then counts down 5 seconds:
# Ctrl-C at any point before the destroy stops it. A box left running costs far more than
# a box destroyed by mistake, whose files are copied first anyway.
# It still asks if something is running on the box, or if the box can't be reached.
# Before each copy it says how big it is, and during a long one, how much has arrived.
# On the box itself it stops at once.
#
#   --keep  copy only; leave the box running
#   --yes   no countdown, and yes to those two questions (for scripted use)
set -euo pipefail

# shellcheck source=scripts/vast/common.sh
. "$(dirname "$0")/common.sh"
laptop_only

DEST="" KEEP=0
for arg in "$@"; do
  case $arg in
    --keep) KEEP=1 ;;
    --yes | -y) YES=1 ;;
    -h | --help)
      awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0" # the header comment
      exit 0
      ;;
    -*) die "unknown option: $arg (try --help)" ;;
    *) DEST=$arg ;;
  esac
done

# Every instance still on the account, as "id(status)"; empty if none.
remaining() {
  vastai show instances --raw 2>&1 | json '
if not isinstance(d, list):
    sys.exit("cannot list instances: %s" % d)
print(" ".join("%s(%s, label %s)" % (i["id"], i.get("actual_status"), i.get("label")) for i in d))'
}

if [ ! -f "$STATE" ]; then
  left=$(remaining) || die "no box recorded in $STATE, and vastai can't list instances"
  if [ -z "$left" ]; then
    say "no box recorded in $STATE, and no instances on the Vast account. Nothing is billing."
    exit 0
  fi
  warn "no box recorded in $STATE, but the account has instances, which are billing: $left"
  warn "if up.sh rented one of them on another computer, run down.sh there: it copies the box's files before destroying it"
  die "otherwise, destroy each one you don't need, without copying anything from it, with: vastai destroy instance <id> -y"
fi
read -r ID _ <"$STATE" || true
[ -n "$ID" ] || die "$STATE is malformed; check https://cloud.vast.ai/instances/"
DEST=${DEST:-$REPO/logs/vast-$(date +%F)-$ID}
billing_traps # until the destroy is confirmed, any stop says the box is still billing

# Before anything slow, so a run started by mistake can be stopped in time.
if [ "$KEEP" = 1 ]; then
  say "--keep: copying from instance $ID; it stays up and keeps billing"
else
  banner "down.sh WILL DESTROY VAST INSTANCE $ID once its files are copied." \
    "Everything else on the box is deleted, and billing stops." "" \
    "To stop it:                 press Ctrl-C (any time before the destroy)" \
    "To copy and keep the box:   scripts/vast/down.sh --keep"
  if [ "$YES" != 1 ]; then
    printf 'starting in' >&2
    for s in 5 4 3 2 1; do
      printf ' %s' "$s" >&2
      sleep 1
    done
    echo >&2
  fi
fi

# Training run bundles come back without optimizer states, prepared datasets and plan's
# work dir: they are large, and plan or a resume rebuilds what is needed. Adapter weights
# do come back: until `fc train upload` has pushed them, the box may hold the only copy.
# RUNS_PRUNE is the same three patterns for find, which sizes the copy first (in find's
# -path, as in tar's --exclude, `*` matches `/` too).
RUNS_EXCLUDE="--exclude=optimizer.pt --exclude='*/prepared' --exclude='*/_candidates'"
RUNS_PRUNE="-name optimizer.pt -o -path '*/prepared' -o -path '*/_candidates'"

say "connecting to the box, and checking nothing is still running on it"
if box true 2>/dev/null; then
  busy=$(box "pgrep -af '$BUSY_PATTERN' || true")
  if [ -n "$busy" ]; then
    warn "still running on the box, and destroying it would kill this:"
    echo "$busy" | cut -c1-160 >&2
    confirm "Copy what is there now and carry on?" || exit 1
  fi
  mkdir -p "$DEST"
  # "<KB> <path>" lines, one per file each copy sends back: files only, as the progress
  # ticker counts them. A failed call stops here, so a dropped connection can't read as
  # "the box has nothing to copy" and go on to destroy it.
  sizes=$(box "cd fragile-compassion 2>/dev/null || exit 0
for d in logs results; do [ ! -d \$d ] || find \$d -type f -exec du -k {} +; done
[ ! -d runs ] || find runs \\( $RUNS_PRUNE \\) -prune -o -type f -exec du -k {} +")
  for dir in logs results runs; do
    kb=$(printf '%s\n' "$sizes" |
      awk -v d="$dir" '$2 == d || index($2, d "/") == 1 {kb += $1; n++} END {if (n) print kb}')
    [ -n "$kb" ] || continue
    if [ "$dir" = runs ]; then
      say "copying runs/, without optimizer states or prepared datasets (about $(size_kb "$kb"))"
    else
      say "copying $dir/ (about $(size_kb "$kb"))"
    fi
    start_progress "$DEST" "$kb"
    if [ "$dir" = runs ]; then
      box "cd fragile-compassion && tar -cf - $RUNS_EXCLUDE runs" | tar -C "$DEST" -xf -
    else
      scp -r -q "${CM[@]}" "vast-em:fragile-compassion/$dir" "$DEST/"
    fi
    stop_progress
  done
  say "saving the box's setup log, and any edits not committed on it"
  scp -q "${CM[@]}" vast-em:/root/fc-setup.log "$DEST/" 2>/dev/null || true
  # Destroying the box deletes any edits made there; keep them as a patch.
  box 'cd fragile-compassion && git diff HEAD' >"$DEST/box-uncommitted.patch" 2>/dev/null || true
  if [ -s "$DEST/box-uncommitted.patch" ]; then
    warn "the box had uncommitted edits; saved them as $DEST/box-uncommitted.patch"
  else
    rm -f "$DEST/box-uncommitted.patch"
  fi
  say "copied to $DEST:"
  du -sh "$DEST"/* 2>/dev/null || echo "   (nothing: the box has no logs/ or results/ yet)"
else
  warn "can't reach the box over SSH, so nothing can be copied from it"
  [ "$KEEP" = 1 ] && exit 1
  confirm "Destroy instance $ID without copying anything?" || exit 1
fi

if [ "$KEEP" = 1 ]; then
  say "not destroyed (--keep)"
  still_billing_banner "$ID"
  exit 0
fi

# `destroy instance` exits 0 and prints nothing useful even when it fails, so ask Vast
# until the instance is really gone.
say "destroying instance $ID"
vastai destroy instance "$ID" -y >/dev/null 2>&1 || true
gone=0
for _ in $(seq 24); do
  if instance_gone "$(instance_row "$ID")"; then
    gone=1
    break
  fi
  sleep 5
done
if [ "$gone" != 1 ]; then
  warn "Vast still lists instance $ID two minutes after the destroy request. It may still be billing."
  die "check https://cloud.vast.ai/instances/ and destroy it there; $STATE is kept until then"
fi

gone_id=$ID
ID="" # gone: from here on, a failure must not say it is still billing
ssh "${CM[@]}" -O exit vast-em >/dev/null 2>&1 || true # close the shared connection
rm -f "$STATE"
echo "# No Vast box right now: scripts/vast/down.sh destroyed instance $gone_id." >"$SNIPPET"
: >"$KNOWN_HOSTS"
say "destroyed instance $gone_id"

left=$(remaining) || left="(could not list)"
if [ -n "$left" ]; then
  warn "the account still has instances, which are billing: $left"
  warn "see https://cloud.vast.ai/instances/"
else
  say "no instances left on the Vast account; nothing is billing"
fi
