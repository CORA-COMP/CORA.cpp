#!/usr/bin/env bash
# build - configures (once) and builds CORA.cpp with CMake, into $CORACPP_BUILD
#
# Syntax:   scripts/with_env.sh scripts/build.sh [--debug] [--eigen] [-DOPTION=value ...] [target ...]
# Example:  scripts/with_env.sh scripts/build.sh                     everything
#           scripts/with_env.sh scripts/build.sh example_zonotope_01 one target (a test, an example, python)
#           scripts/with_env.sh scripts/build.sh --debug example_zonotope_01   -O0 -g into <build>-debug
#           scripts/with_env.sh scripts/build.sh --eigen              without libtorch
#           scripts/with_env.sh scripts/build.sh -DCORACPP_ORACLE=<dir>/oracle-unsound
#                                                         the tests against another CORALean oracle (CONTRIBUTING.md)
# The build directory is $CORACPP_BUILD (scripts/with_env.sh sets it: .coracpp-build, else ./build).
# Run the tests with: scripts/with_env.sh scripts/test.sh -j4
#
# Authors:       Claude Sonnet 5.5, Claude Opus 5.5
# Written:       30-September-2026
# Last update:   18-November-2026
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${CORACPP_BUILD:-$ROOT/build}"
TYPE=Release
ARGS=()
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --debug) TYPE=Debug ;;
        --eigen) ARGS+=("-DCORACPP_TORCH=OFF") ;;
        -D*) ARGS+=("$arg") ;;
        -*) echo "build: unknown option '$arg'; use --debug, --eigen, -DOPTION=value or a target name" >&2; exit 2 ;;
        *) TARGETS+=("$arg") ;;
    esac
done
if [ "$TYPE" = Debug ]; then BUILD="$BUILD-debug"; fi

# A fresh build directory gets the build type, ninja and ccache (when there are) once; an existing
# one keeps its own settings, so building in it, as the Python tests do, changes nothing but targets
CONFIGURE=()
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    CONFIGURE=(-DCMAKE_BUILD_TYPE="$TYPE")
    if command -v ninja >/dev/null 2>&1; then CONFIGURE+=(-G Ninja); fi
    if command -v ccache >/dev/null 2>&1; then CONFIGURE+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache); fi
fi

# configure a fresh directory or apply new -D options; else the build reruns CMake when it must.
# The configure output is shown only when it fails; the build prints its own progress
if [ ${#CONFIGURE[@]} -gt 0 ] || [ ${#ARGS[@]} -gt 0 ]; then
    if ! LOG="$(cmake -S "$ROOT" -B "$BUILD" ${CONFIGURE[@]+"${CONFIGURE[@]}"} \
            ${ARGS[@]+"${ARGS[@]}"} 2>&1)"; then
        echo "$LOG" >&2
        exit 1
    fi
fi
# a number of jobs: a bare --parallel means unlimited with make, and libtorch compiles run out of memory
JOBS="${CORACPP_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
if [ ${#TARGETS[@]} -gt 0 ]; then
    cmake --build "$BUILD" --parallel "$JOBS" --target "${TARGETS[@]}"
else
    cmake --build "$BUILD" --parallel "$JOBS"
fi

# ----------------------------------------  END OF CODE  ----------------------------------------- #
