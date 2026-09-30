#!/usr/bin/env bash
# build - configures (once) and builds CORA.cpp with CMake, into $CORACPP_BUILD
#
# Syntax:   scripts/with_env.sh scripts/build.sh [--debug] [--eigen] [-DOPTION=value ...] [target ...]
# Example:  scripts/with_env.sh scripts/build.sh                     everything
#           scripts/with_env.sh scripts/build.sh example_zonotope_01 one target (a test, an example, python)
#           scripts/with_env.sh scripts/build.sh --debug example_zonotope_01   -O0 -g into <build>-debug
#           scripts/with_env.sh scripts/build.sh --eigen              without libtorch
# The build directory is $CORACPP_BUILD (scripts/with_env.sh sets it: .coracpp-build, else ./build).
# Run the tests with: scripts/with_env.sh scripts/test.sh -j4

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

# ccache, when there is one, keeps a rebuild after a clean cheap; ninja is faster than make
LAUNCHER=()
GENERATOR=()
if command -v ccache >/dev/null 2>&1; then LAUNCHER=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache); fi
# a generator is chosen once: an existing build directory keeps its own
if [ ! -f "$BUILD/CMakeCache.txt" ] && command -v ninja >/dev/null 2>&1; then GENERATOR=(-G Ninja); fi

# the configure output is shown only when it fails; the build prints its own progress
if ! LOG="$(cmake -S "$ROOT" -B "$BUILD" ${GENERATOR[@]+"${GENERATOR[@]}"} -DCMAKE_BUILD_TYPE="$TYPE" \
        ${LAUNCHER[@]+"${LAUNCHER[@]}"} ${ARGS[@]+"${ARGS[@]}"} 2>&1)"; then
    echo "$LOG" >&2
    exit 1
fi
# a number of jobs: a bare --parallel means unlimited with make, and libtorch compiles run out of memory
JOBS="${CORACPP_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
if [ ${#TARGETS[@]} -gt 0 ]; then
    cmake --build "$BUILD" --parallel "$JOBS" --target "${TARGETS[@]}"
else
    cmake --build "$BUILD" --parallel "$JOBS"
fi
