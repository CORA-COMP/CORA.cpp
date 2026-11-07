# oracle - the CORALean oracle that the lean tests, examples and Python package run against
# Included by CMakeLists.txt. CORACPP_ORACLE is OFF, DOWNLOAD (the release pinned in
# scripts/coralean.cmake) or a command that starts the oracle; the tests skip what needs it when OFF.

set(CORACPP_ORACLE "OFF" CACHE STRING "The CORALean oracle: OFF, DOWNLOAD, or the command that starts it")

# The command of the oracle, fetching the pinned release if asked to.
function(cora_oracle_command out)
  set(setting "${CORACPP_ORACLE}")
  if(setting STREQUAL "DOWNLOAD")
    include("${CMAKE_SOURCE_DIR}/scripts/coralean.cmake")
    if(CORALEAN_VERSION STREQUAL "unpublished" OR CORALEAN_ARCHIVE_SHA256 STREQUAL "")
      message(FATAL_ERROR "CORACPP_ORACLE=DOWNLOAD: no CORALean oracle is pinned yet; "
                          "point CORACPP_ORACLE at a local build instead (CONTRIBUTING.md)")
    endif()
    # a present file with the pinned hash is not fetched again
    set(directory "${CMAKE_BINARY_DIR}/coralean/${CORALEAN_VERSION}")
    set(archive "${directory}/${CORALEAN_ARCHIVE}")
    file(DOWNLOAD "${CORALEAN_URL}/${CORALEAN_ARCHIVE}" "${archive}"
         EXPECTED_HASH SHA256=${CORALEAN_ARCHIVE_SHA256} SHOW_PROGRESS)
    set(file "${directory}/${CORALEAN_ORACLE_ASSET}")
    if(NOT EXISTS "${file}")
      file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${directory}")
    endif()
    file(CHMOD "${file}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    set(setting "${file}")
  endif()
  set(${out} "${setting}" PARENT_SCOPE)
endfunction()

set(coracpp_oracle_env "")
if(NOT CORACPP_ORACLE STREQUAL "OFF")
  cora_oracle_command(command)
  set(coracpp_oracle_env "CORACPP_ORACLE=${command}")
  message(STATUS "CORALean oracle: ${command}")
endif()
