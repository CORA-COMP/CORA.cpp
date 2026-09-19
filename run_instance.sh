#!/bin/bash

# run_instance.sh — run one instance and report the verdict.
#   args: v1 <benchmark> <instance> <params> [further catalog columns...] <results file>
#
# Everything here is timed, so it only hands params to the warm daemon (started by
# prepare_instance.sh) and waits for the verdict. The daemon runs the whole instance:
# generate the inputs, then repeat the operation params.repetition times. Without a
# daemon, the instance runs in a fresh process instead.

set -u

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/coracpp_lib.sh"

PARAMS="$4"
# The results file is always the last argument.
OUT="${@: -1}"

ask "run	$OUT	$PARAMS"
case $? in
    0)  # The daemon's log only matters when the instance did not finish.
        [ "$REPLY" = finished ] || cat "$SRV_DIR/job.log"
        exit 0 ;;
    1)  echo "[run] no CORA.cpp daemon; running directly"
        exec "$CORACPP" run "$PARAMS" "$OUT" ;;
    *)  echo "[run] the CORA.cpp daemon died during the instance"
        cat "$SRV_DIR/job.log"
        exit 1 ;;
esac
