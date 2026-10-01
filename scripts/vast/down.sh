#!/usr/bin/env bash
# Copy results off the rented Vast.ai box, then destroy it. Runs on your laptop.
#
#   scripts/vast/down.sh [--yes] [dest-dir]      (default: logs/vast-<date>-<instance>/)
#
# Copies the box's logs/ and results/ (plus the setup log, and a patch of any edits
# made on the box) into dest-dir, asks, destroys the instance, and waits until Vast
# confirms it is gone. Answer "no" to the destroy question to copy only and keep the box.
#
#   --yes   answer yes to every question (for scripted use)
set -euo pipefail

# shellcheck source=scripts/vast/common.sh
. "$(dirname "$0")/common.sh"

DEST=""
for arg in "$@"; do
  case $arg in
    --yes | -y) YES=1 ;;
    -h | --help)
      sed -n '2,10p' "$0"
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
  die "destroy each one you don't need with: vastai destroy instance <id> -y"
fi
read -r ID _ <"$STATE" || true
[ -n "$ID" ] || die "$STATE is malformed; check https://cloud.vast.ai/instances/"
DEST=${DEST:-$REPO/logs/vast-$(date +%F)-$ID}

if box true 2>/dev/null; then
  busy=$(box 'pgrep -af "[f]c run|[i]nspect eval|[v]llm serve" || true')
  if [ -n "$busy" ]; then
    warn "still running on the box:"
    echo "$busy" | cut -c1-160 >&2
    confirm "Copy what is there now and carry on?" || exit 1
  fi
  mkdir -p "$DEST"
  for dir in logs results; do
    if box "test -d fragile-compassion/$dir"; then
      scp -r -q "${CM[@]}" "vast-em:fragile-compassion/$dir" "$DEST/"
    fi
  done
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
  confirm "Destroy instance $ID without copying anything?" || exit 1
fi

if ! confirm "Destroy instance $ID? This deletes everything on it."; then
  say "not destroyed. Instance $ID is still billing; run down.sh again when you're done."
  exit 0
fi

# `destroy instance` exits 0 and prints nothing useful even when it fails, so ask Vast
# until the instance is really gone.
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

ssh "${CM[@]}" -O exit vast-em >/dev/null 2>&1 || true # close the shared connection
rm -f "$STATE"
echo "# No Vast box right now: scripts/vast/down.sh destroyed instance $ID." >"$SNIPPET"
: >"$KNOWN_HOSTS"
say "destroyed instance $ID"

left=$(remaining) || left="(could not list)"
if [ -n "$left" ]; then
  warn "the account still has instances, which are billing: $left"
  warn "see https://cloud.vast.ai/instances/"
else
  say "no instances left on the Vast account; nothing is billing"
fi
