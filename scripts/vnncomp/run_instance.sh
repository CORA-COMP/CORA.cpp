#!/bin/bash
# run_instance - one VNN-COMP instance: the verdict and the counterexample into the result file
#
# Syntax:   run_instance.sh v1 <category> <onnx> <vnnlib> <result file> <timeout>
#
# VNN-COMP's tool interface (run_single_instance.sh of vnncomp2026_benchmarks). The Python of
# CORACPP_VNNCOMP_PYTHON (default python3) runs scripts/vnncomp/run_instance.py with the cora
# package of CORACPP_BUILD (default build/python-torch) on PYTHONPATH; the timeout counts from here.
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

export VNNCOMP_START="$(date +%s.%N)"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="${CORACPP_BUILD:-$ROOT/build/python-torch}${PYTHONPATH:+:$PYTHONPATH}"
exec "${CORACPP_VNNCOMP_PYTHON:-python3}" "$HERE/run_instance.py" "$2" "$3" "$4" "$5" "$6"

# ----------------------------------------  END OF CODE  ----------------------------------------- #
