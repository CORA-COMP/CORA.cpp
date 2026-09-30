#!/usr/bin/env bash
# test - runs the tests of the build (conventions, C++ tests, examples, Python) with ctest
#
# Syntax:   scripts/with_env.sh scripts/test.sh [ctest options]
# Example:  scripts/with_env.sh scripts/test.sh -j4           everything
#           scripts/with_env.sh scripts/test.sh -L example    only the C++ examples (labels: test, example, python)
# The build directory is $CORACPP_BUILD (scripts/with_env.sh sets it), else ./build.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec ctest --test-dir "${CORACPP_BUILD:-$ROOT/build}" --output-on-failure "$@"
