#!/bin/bash
# coracpp_lib.sh — shared by the tool scripts; builtins only in ask(), since
# run_instance.sh calls it inside the measured region.
#
# Sets: HERE (the repository), CORACPP (the binary), SRV_DIR, PORT; ask() sets REPLY.
# The tool scripts run under `set -u`, so nothing here may read an unset variable.

: "${HERE:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CORACPP="${CORACPP_BIN:-$HERE/build/coracpp}"
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
