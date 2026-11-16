#!/bin/bash
# coracpp_lib - shared by the tool scripts; builtins only in ask(), since run_instance.sh calls it in the measured region
#
# Sets: HERE (this directory), ROOT (the repository), CORACPP (the binary), SRV_DIR, PORT;
# ask() sets REPLY.
# The tool scripts run under `set -u`, so nothing here may read an unset variable.
#
# Authors:       Claude Sonnet 5.5
# Written:       01-October-2026
# Last update:   ---
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

: "${HERE:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="$(cd "$HERE/../.." && pwd)"
# Where install_tool.sh found libtorch, when it did: the binary's rpath covers libtorch's
# own directory but not the NVIDIA libraries a pip torch keeps beside it.
if [ -s "$ROOT/.libtorch-path" ]; then
    export LD_LIBRARY_PATH="$(cat "$ROOT/.libtorch-path"):${LD_LIBRARY_PATH:-}"
fi
CORACPP="${CORACPP_BIN:-$ROOT/build/coracpp}"
SRV_DIR="${CORACPP_SERVER_DIR:-${HOME:-/tmp}/.coracpp_server}"
PORT="${CORACPP_PORT:-47916}"
export CORACPP_PORT="$PORT"

# ask REQUEST [TIMEOUT]: send one request line to the daemon and read its one-line reply
# into REPLY. Returns 1 if no daemon is listening, 2 if it did not answer.
ask() {
    REPLY=""
    { exec 3<>"/dev/tcp/127.0.0.1/$PORT"; } 2>/dev/null || return 1
    printf '%s\n' "$1" >&3
    if ! read -r ${2:+-t "$2"} REPLY <&3 2>/dev/null; then
        exec 3>&-
        return 2
    fi
    exec 3>&-
    return 0
}

# ----------------------------------------  END OF CODE  ----------------------------------------- #
