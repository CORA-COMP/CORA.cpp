#!/bin/bash

# install_tool.sh — run once on the worker to install CORA.cpp.
#
# Installs a compiler, Eigen and GLPK if the image lacks them, builds the tool, and runs
# every operation once, so a broken setup fails the install rather than the instances.
#
# Argument:
# - $1: interface version string, e.g. "v1"

set -e

VERSION="${1:-v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/coracpp_lib.sh"
echo "Installing CORA.cpp (interface $VERSION)"

missing=""
command -v g++ >/dev/null || missing="$missing g++"
command -v make >/dev/null || missing="$missing make"
[ -d "${EIGEN:-/usr/include/eigen3}" ] || missing="$missing libeigen3-dev"
[ -f /usr/include/glpk.h ] || missing="$missing libglpk-dev"

if [ -n "$missing" ]; then
    # The platform runs the installation as the same user as the instances, so apt needs
    # the node's passwordless sudo.
    if [ "$(id -u)" -eq 0 ]; then
        as_root=""
    elif sudo -n true 2>/dev/null; then
        as_root="sudo -n"
    else
        echo "missing:$missing — install them, or give the installing user root or sudo"
        exit 1
    fi
    echo "installing:$missing"
    $as_root apt-get update -qq
    # shellcheck disable=SC2086
    $as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing >/dev/null
fi

g++ --version | head -1
# Separate invocations: in one parallel make, `clean` would race the compiles it precedes.
make -C "$HERE" clean
make -C "$HERE" -j"$(nproc)" all

# What the worker will actually run on, and every operation once.
"$CORACPP" env
"$CORACPP" check

# The instances may run as another user than the install.
chmod -R a+rX "$HERE"
