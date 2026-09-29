#!/usr/bin/env bash
# with_env - runs a command inside the coracpp conda environment
#
# Syntax:   scripts/with_env.sh <command> [arguments...]
# Example:  scripts/with_env.sh make test example
#           scripts/with_env.sh make python && scripts/with_env.sh python examples/python/example_linear_reach_01_5dim.py
# It also puts the built `cora` package on PYTHONPATH.
# The environment comes from scripts/setup_local.sh; CORACPP_ENV names another one, and
# CORACPP_ENV=none runs in the current environment (CI, containers) with only the paths below set.
#
# The environment is entered by hand rather than with `conda activate`: its bin first on PATH, its
# lib first on the library path (torch would otherwise load the system's older libstdc++ and
# numpy could not), and its compiler as CXX.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${CORACPP_ENV:-coracpp}"

if [ "$ENV_NAME" != "none" ]; then
    # The environment's prefix: a directory of a conda installation, else what `conda env list` of any
    # conda found says (the shell of an editor may have another conda first on PATH).
    PREFIX=""
    for base in "${CONDA_ROOT:-}" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"             "$HOME/.conda" "/opt/conda"; do
        for dir in "$base/envs/$ENV_NAME" "$base/$ENV_NAME"; do
            if [ -n "$base" ] && [ -x "$dir/bin/python" ]; then PREFIX="$dir"; break 2; fi
        done
    done
    if [ -z "$PREFIX" ]; then
        for candidate in "$(command -v conda || true)" "$(command -v mamba || true)" "$HOME/miniforge3/bin/conda"; do
            [ -n "$candidate" ] && [ -x "$candidate" ] || continue
            found="$("$candidate" env list 2>/dev/null | awk -v name="$ENV_NAME" '$1 == name {print $NF}')"
            if [ -n "$found" ] && [ -d "$found" ]; then PREFIX="$found"; break; fi
        done
    fi
    if [ -z "$PREFIX" ]; then
        echo "with_env: no environment '$ENV_NAME' found in the usual conda locations or by conda; run scripts/setup_local.sh first (CORACPP_ENV names another environment, CONDA_ROOT another installation)" >&2
        exit 1
    fi

    export CONDA_PREFIX="$PREFIX"
    export PATH="$PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -x "$PREFIX/bin/x86_64-conda-linux-gnu-c++" ]; then export CXX="$PREFIX/bin/x86_64-conda-linux-gnu-c++"; fi
fi

cd "$ROOT"

# The Python package is built into the build directory (local.mk, else ./build), so put it on
# the path: python scripts and examples find `cora` without further setup. Without a local.mk
# (a fresh checkout, CI) that is ./build.
BUILD_DIR="$( { sed -n 's/^BUILD *:\?= *//p' local.mk 2>/dev/null || true; } | head -1 | sed "s|\$(HOME)|$HOME|g")"
export CORACPP_BUILD="${BUILD_DIR:-$ROOT/build}"
export PYTHONPATH="$CORACPP_BUILD${PYTHONPATH:+:$PYTHONPATH}"

exec "$@"
