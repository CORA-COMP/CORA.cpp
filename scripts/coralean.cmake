# The CORALean oracle that CORACPP_ORACLE=DOWNLOAD fetches (Linux x86_64), read by oracle.cmake.
# CORALean's CI publishes it on main as a tag of the public mirror (its scripts/build/github-oracle.sh).
# Of the two executables in the archive CORA.cpp runs the unsound one: it serves every dtype, nearest
# included. Bump the tag and the hash together, to the values of that job's
# `oracle-linux-x86_64.tar.xz.sha256`.
set(CORALEAN_VERSION "oracle-20261113-313c5ebf6")
set(CORALEAN_URL "https://github.com/TUMcps/CORALean/raw/${CORALEAN_VERSION}")
set(CORALEAN_ARCHIVE "oracle-linux-x86_64.tar.xz")
set(CORALEAN_ARCHIVE_SHA256 "03ec31586442afe2713c5f6ea99ff3143025ce810a43b69d7aa1e606a7f23b26")
# the executable inside the archive that CORACPP_ORACLE runs
set(CORALEAN_ORACLE_ASSET "oracle-unsound-linux-x86_64")
