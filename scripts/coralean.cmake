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

set(CORALEAN_VERSION "oracle-20261124-0322eafde")
set(CORALEAN_URL "https://github.com/TUMcps/CORALean/raw/${CORALEAN_VERSION}")
set(CORALEAN_ARCHIVE "oracle-linux-x86_64.tar.xz")
set(CORALEAN_ARCHIVE_SHA256 "c67338adc4f6c9f9840388943b417a8bc906ee16b6e0db2c3698a38351f0b054")
# the executable inside the archive that CORACPP_ORACLE runs
set(CORALEAN_ORACLE_ASSET "oracle-unsound-linux-x86_64")

# ----------------------------------------  END OF CODE  ----------------------------------------- #
