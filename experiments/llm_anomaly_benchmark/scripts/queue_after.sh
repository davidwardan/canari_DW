#!/usr/bin/env bash
# Wait for a running process to exit, then run the given command.
#   scripts/queue_after.sh <pid> <command...>
pid="$1"; shift
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] pid $pid exited; starting: $*"
exec "$@"
