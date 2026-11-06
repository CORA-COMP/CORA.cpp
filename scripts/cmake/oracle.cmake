# oracle - the CORALean oracle that the lean tests, examples and Python package run against
# Included by CMakeLists.txt. CORACPP_ORACLE is OFF, DOWNLOAD (the release pinned in
# scripts/coralean.cmake) or a command that starts the oracle; the tests skip what needs it when OFF.
# CORACPP_ORACLE_UNSOUND is the same for the unsound oracle (dtype nearest, no guarantees).

set(CORACPP_ORACLE "OFF" CACHE STRING "The CORALean oracle: OFF, DOWNLOAD, or the command that starts it")
set(CORACPP_ORACLE_UNSOUND "OFF" CACHE STRING "The CORALean unsound oracle: OFF, DOWNLOAD, or a command")

# The command for `which` (ORACLE or ORACLE_UNSOUND), fetching the pinned release if asked to.
function(cora_oracle_command which out)
  set(setting "${CORACPP_${which}}")
  if(setting STREQUAL "DOWNLOAD")
    include("${CMAKE_SOURCE_DIR}/scripts/coralean.cmake")
    if(CORALEAN_VERSION STREQUAL "unpublished" OR CORALEAN_ARCHIVE_SHA256 STREQUAL "")
      message(FATAL_ERROR "CORACPP_${which}=DOWNLOAD: no CORALean oracle is pinned yet; "
                          "point CORACPP_${which} at a local build instead (CONTRIBUTING.md)")
    endif()
    # One archive for both oracles; a present file with the pinned hash is not fetched again.
    set(directory "${CMAKE_BINARY_DIR}/coralean/${CORALEAN_VERSION}")
    set(archive "${directory}/${CORALEAN_ARCHIVE}")
    file(DOWNLOAD "${CORALEAN_URL}/${CORALEAN_ARCHIVE}" "${archive}"
         EXPECTED_HASH SHA256=${CORALEAN_ARCHIVE_SHA256} SHOW_PROGRESS)
    set(file "${directory}/${CORALEAN_${which}_ASSET}")
    if(NOT EXISTS "${file}")
      file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${directory}")
    endif()
    file(CHMOD "${file}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    set(setting "${file}")
  endif()
  set(${out} "${setting}" PARENT_SCOPE)
endfunction()

set(coracpp_oracle_env "")
foreach(which ORACLE ORACLE_UNSOUND)
  if(NOT CORACPP_${which} STREQUAL "OFF")
    cora_oracle_command(${which} command)
    list(APPEND coracpp_oracle_env "CORACPP_${which}=${command}")
    message(STATUS "CORALean ${which}: ${command}")
  endif()
endforeach()
