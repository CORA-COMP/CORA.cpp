#!/bin/bash
# prepare_instance - the untimed check before an instance: the network and the property load
#
# Syntax:   prepare_instance.sh v1 <category> <onnx> <vnnlib>
#
# VNN-COMP's tool interface. A file that does not load is no reason to fail the instance (the run
# then answers unknown), so this exits 0 unless the arguments are wrong.
#
# Authors:       Claude Opus 5.5
# Written:       21-November-2026
# Last update:   ---
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

set -u

if [ "$1" != "v1" ]; then
    echo "Expected first argument (version string) 'v1', got '$1'"
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="${CORACPP_BUILD:-$ROOT/build/python-torch}${PYTHONPATH:+:$PYTHONPATH}"
"${CORACPP_VNNCOMP_PYTHON:-python3}" "$HERE/run_instance.py" --prepare "$2" "$3" "$4" \
    || echo "[prepare] the instance does not load; run_instance.sh will answer unknown"
exit 0

# ----------------------------------------  END OF CODE  ----------------------------------------- #
