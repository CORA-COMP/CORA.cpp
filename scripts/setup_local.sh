#!/usr/bin/env bash
# setup_local - creates the environment CORA.cpp builds and runs in, without root and Docker
#
# Syntax:   scripts/setup_local.sh [--cpu]
# What it does: finds (or installs, into ~/miniforge3) conda; creates the environment `coracpp`
#           from environment.yml; installs torch with pip (the CUDA build if there is an NVIDIA
#           GPU, else the CPU one, or --cpu to force it); and, when the repository is on a
#           slow drive (WSL's /mnt/c), writes local.mk so that objects are built elsewhere.
# Afterwards: scripts/with_env.sh make test        runs a command inside the environment
#
# Linux and WSL2 only. Set CORACPP_ENV to use another environment name.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${CORACPP_ENV:-coracpp}"
FORCE_CPU=0
[ "${1:-}" = "--cpu" ] && FORCE_CPU=1

echo "[setup] repository: $ROOT"

# --- conda -----------------------------------------------------------------------------
if command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA="$HOME/miniforge3/bin/conda"
else
    echo "[setup] installing Miniforge into ~/miniforge3"
    installer="$(mktemp --suffix=.sh)"
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o "$installer"
    bash "$installer" -b -p "$HOME/miniforge3"
    rm -f "$installer"
    CONDA="$HOME/miniforge3/bin/conda"
fi
echo "[setup] using $CONDA"

# --- the environment ---------------------------------------------------------------------
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] updating environment $ENV_NAME"
    "$CONDA" env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
else
    echo "[setup] creating environment $ENV_NAME"
    "$CONDA" env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi

# --- torch -------------------------------------------------------------------------------
if [ "$FORCE_CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    index="https://download.pytorch.org/whl/cu126"
    echo "[setup] installing torch (CUDA 12.6)"
else
    index="https://download.pytorch.org/whl/cpu"
    echo "[setup] installing torch (CPU)"
fi
"$CONDA" run -n "$ENV_NAME" --no-capture-output python -m pip install --quiet torch --index-url "$index"

# --- a fast build directory when the sources are on a slow drive ------------------------
case "$ROOT" in
    /mnt/*)
        printf 'BUILD := $(HOME)/.cache/coracpp/build\n' > "$ROOT/local.mk"
        echo "[setup] the repository is on a Windows drive: objects go to ~/.cache/coracpp/build (local.mk)"
        ;;
esac

"$ROOT/scripts/with_env.sh" python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] done. Try: scripts/with_env.sh make test example"
