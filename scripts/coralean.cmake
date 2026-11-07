# The CORALean oracle that CORACPP_ORACLE=DOWNLOAD fetches (Linux x86_64), read by oracle.cmake.
# CORALean's CI publishes it on main as a tag of the public mirror (its scripts/build/github-oracle.sh):
# one archive with the sound `oracle` and the unsound one that offers nearest. Bump the tag and the hash
# together, to the values of that job's `oracle-linux-x86_64.tar.xz.sha256`.
set(CORALEAN_VERSION "oracle-20261107-9c0c5b775")
set(CORALEAN_URL "https://github.com/TUMcps/CORALean/raw/${CORALEAN_VERSION}")
set(CORALEAN_ARCHIVE "oracle-linux-x86_64.tar.xz")
set(CORALEAN_ARCHIVE_SHA256 "4d3497337306d5e9b0d5b5f8915ec34497004297168f4f1293534dcc38aa8a61")
# the executables inside the archive
set(CORALEAN_ORACLE_ASSET "oracle-linux-x86_64")
set(CORALEAN_ORACLE_UNSOUND_ASSET "oracle-unsound-linux-x86_64")
