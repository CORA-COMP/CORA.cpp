#!/bin/bash

# install_tool.sh — run once on the worker to install CORA.cpp.
#
# Installs a compiler, Eigen and GLPK if the image lacks them, builds the tool, and runs
# every operation once, so a broken setup fails the install rather than the instances.
#
# Builds the libtorch backend in whenever libtorch can be found — $LIBTORCH, else the
# image's Python torch, which is where a pytorch/pytorch image keeps the C++ headers and
# a CUDA build. Without it the tool is the Eigen CPU one and gpu instances report
# unsupported, so an image without torch still installs.
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

# libtorch: $LIBTORCH wins, else whatever the image's Python torch unpacked.
torch_dir="${LIBTORCH:-}"
torch_abi=1
if [ -z "$torch_dir" ]; then
    for py in "${CORACPP_PYTHON:-}" /opt/conda/bin/python3 python3 python; do
        [ -n "$py" ] && command -v "$py" >/dev/null 2>&1 || continue
        torch_dir="$("$py" -c 'import os, torch; print(os.path.dirname(torch.__file__))' \
                     2>/dev/null)" || torch_dir=""
        if [ -n "$torch_dir" ]; then
            torch_abi="$("$py" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' \
                         2>/dev/null || echo 1)"
            break
        fi
    done
fi
# A pip torch keeps the C++ API's headers under csrc, not beside the C10 ones.
if [ -n "$torch_dir" ] \
   && [ ! -f "$torch_dir/include/torch/csrc/api/include/torch/torch.h" ]; then
    echo "libtorch at $torch_dir has no C++ headers; building without it"
    torch_dir=""
fi
if [ -n "$torch_dir" ]; then
    echo "libtorch: $torch_dir (cxx11 abi $torch_abi)"
else
    echo "libtorch: not found — building the Eigen CPU backend only"
fi
# The daemon has to find libtorch's shared objects, and the rpath covers only the
# directory itself, not the NVIDIA libraries a pip torch keeps beside it.
: > "$HERE/.libtorch-path"
if [ -n "$torch_dir" ]; then
    printf '%s' "$torch_dir/lib" > "$HERE/.libtorch-path"
    for nvidia in "$torch_dir"/../nvidia/*/lib; do
        [ -d "$nvidia" ] && printf ':%s' "$(cd "$nvidia" && pwd)" >> "$HERE/.libtorch-path"
    done
fi

g++ --version | head -1
# Separate invocations: in one parallel make, `clean` would race the compiles it precedes.
make -C "$HERE" clean
make -C "$HERE" -j"$(nproc)" TORCH="$torch_dir" TORCH_ABI="$torch_abi" all

# What the worker will actually run on, and every operation once.
"$CORACPP" env
"$CORACPP" check

# The instances may run as another user than the install.
chmod -R a+rX "$HERE"
