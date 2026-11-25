# coralean - the CORALean oracle that CORACPP_ORACLE=DOWNLOAD fetches (Linux x86_64), read by oracle.cmake
#
# CORALean's CI publishes it on main as a tag of the public mirror (its scripts/build/github-oracle.sh).
# Of the two executables in the archive CORA.cpp runs the unsound one: it serves every dtype, nearest
# included. Bump the tag and the hash together, to the values of that job's
# `oracle-linux-x86_64.tar.xz.sha256`.
#
# Authors:       Claude Sonnet 5.5
# Written:       30-September-2026
# Last update:   ---
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

set(CORALEAN_VERSION "oracle-20261125-c8e492f21")
set(CORALEAN_URL "https://github.com/TUMcps/CORALean/raw/${CORALEAN_VERSION}")
set(CORALEAN_ARCHIVE "oracle-linux-x86_64.tar.xz")
set(CORALEAN_ARCHIVE_SHA256 "299c9e6c390c98a4966cc639a0b88137a5a95c40668d8c80e44d26c6960d6d15")
# the executable inside the archive that CORACPP_ORACLE runs
set(CORALEAN_ORACLE_ASSET "oracle-unsound-linux-x86_64")

# ----------------------------------------  END OF CODE  ----------------------------------------- #
