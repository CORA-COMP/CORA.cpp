#!/usr/bin/env bash
# instruction_counts - builds the library at a base commit and at HEAD, and compares the instruction
# counts of benchmarks/micro between them (scripts/ci/instruction_counts.py)
#
# Syntax:   scripts/ci/instruction_counts.sh <base commit> [threshold in percent]
# Needs:    valgrind (or perf), cmake, a compiler; ccache is used when found. Work goes to build/ic-*.
# Both builds use the micro-benchmarks of HEAD, linked against each commit's install tree.

set -euo pipefail

BASE="${1:?usage: instruction_counts.sh <base commit> [threshold]}"
THRESHOLD="${2:-3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$ROOT/build/ic"
LAUNCHER=()
if command -v ccache >/dev/null; then LAUNCHER=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache); fi

# the library of the source tree $1, installed to $WORK/$2, and the micro-benchmarks of HEAD against it
build() {
    local source="$1" name="$2"
    cmake -S "$source" -B "$WORK/$name-lib" -DCMAKE_BUILD_TYPE=Release -DCORACPP_NATIVE=OFF \
        -DCORACPP_TORCH=OFF -DCORACPP_PYTHON=OFF "${LAUNCHER[@]}" >/dev/null
    cmake --build "$WORK/$name-lib" --target cora --parallel
    cmake --install "$WORK/$name-lib" --prefix "$WORK/$name-prefix" >/dev/null
    cmake -S "$ROOT/benchmarks/micro" -B "$WORK/$name-micro" -DCMAKE_PREFIX_PATH="$WORK/$name-prefix" >/dev/null
    cmake --build "$WORK/$name-micro"
}

rm -rf "$WORK"
git -C "$ROOT" worktree prune
mkdir -p "$WORK"
trap 'git -C "$ROOT" worktree remove --force "$WORK/base-source" 2>/dev/null || true' EXIT
git -C "$ROOT" worktree add --detach "$WORK/base-source" "$BASE" >/dev/null
if [ ! -f "$WORK/base-source/scripts/cmake/install.cmake" ]; then
    echo "instruction_counts: $BASE has no install tree (find_package(CORA)); nothing to compare with"
    exit 0
fi
build "$WORK/base-source" base
build "$ROOT" head
python3 "$ROOT/scripts/ci/instruction_counts.py" --threshold "$THRESHOLD" \
    --base "$WORK/base-micro/micro" --head "$WORK/head-micro/micro"
