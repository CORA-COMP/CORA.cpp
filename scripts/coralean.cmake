# The CORALean oracle that CORACPP_ORACLE=DOWNLOAD fetches (Linux x86_64), read by oracle.cmake.
# CORALean's CI publishes it on main as a tag of the public mirror (its scripts/build/github-oracle.sh).
# Of the two executables in the archive CORA.cpp runs the unsound one: it serves every dtype, nearest
# included. Bump the tag and the hash together, to the values of that job's
# `oracle-linux-x86_64.tar.xz.sha256`.
set(CORALEAN_VERSION "oracle-20261109-b09283d9c")
set(CORALEAN_URL "https://github.com/TUMcps/CORALean/raw/${CORALEAN_VERSION}")
set(CORALEAN_ARCHIVE "oracle-linux-x86_64.tar.xz")
set(CORALEAN_ARCHIVE_SHA256 "2b3c3ce6c2b9834bc07431b6157e0ba01a3844398313c37174b8c066d7cc6a14")
# the executable inside the archive that CORACPP_ORACLE runs
set(CORALEAN_ORACLE_ASSET "oracle-unsound-linux-x86_64")
