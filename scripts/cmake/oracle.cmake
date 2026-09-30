# oracle - the CORALean oracle that the lean tests, examples and Python package run against
# Included by CMakeLists.txt. CORACPP_ORACLE is OFF, DOWNLOAD (the release pinned in
# scripts/coralean.cmake) or a command that starts the oracle; the tests skip what needs it when OFF.
# CORACPP_ORACLE_BENCH is the same for the bench oracle (dtypes float and nearest).

set(CORACPP_ORACLE "OFF" CACHE STRING "The CORALean oracle: OFF, DOWNLOAD, or the command that starts it")
set(CORACPP_ORACLE_BENCH "OFF" CACHE STRING "The CORALean bench oracle: OFF, DOWNLOAD, or a command")

# The command for `which` (ORACLE or ORACLE_BENCH), fetching the pinned release if asked to.
function(cora_oracle_command which out)
  set(setting "${CORACPP_${which}}")
  if(setting STREQUAL "DOWNLOAD")
    include("${CMAKE_SOURCE_DIR}/scripts/coralean.cmake")
    if(CORALEAN_VERSION STREQUAL "unpublished" OR "${CORALEAN_${which}_SHA256}" STREQUAL "")
      message(FATAL_ERROR "CORACPP_${which}=DOWNLOAD: CORALean has no published release yet; "
                          "point CORACPP_${which} at a local build instead (CONTRIBUTING.md)")
    endif()
    set(file "${CMAKE_BINARY_DIR}/coralean/${CORALEAN_${which}_ASSET}")
    file(DOWNLOAD "${CORALEAN_URL}/${CORALEAN_${which}_ASSET}" "${file}"
         EXPECTED_HASH SHA256=${CORALEAN_${which}_SHA256} SHOW_PROGRESS)
    file(CHMOD "${file}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    set(setting "${file}")
  endif()
  set(${out} "${setting}" PARENT_SCOPE)
endfunction()

set(coracpp_oracle_env "")
foreach(which ORACLE ORACLE_BENCH)
  if(NOT CORACPP_${which} STREQUAL "OFF")
    cora_oracle_command(${which} command)
    list(APPEND coracpp_oracle_env "CORACPP_${which}=${command}")
    message(STATUS "CORALean ${which}: ${command}")
  endif()
endforeach()
