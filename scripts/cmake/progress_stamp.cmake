# progress_stamp - prints a build report line when something it covers was rebuilt
#
# Run by the custom targets of scripts/cmake/library.cmake: cmake -DCOVERS_FILE=<file>
# -DSTAMP=<file> -DLABEL=<text> [-DBOLD=ON] -P progress_stamp.cmake. COVERS_FILE lists, one per line,
# the object files or the stamps of the folders below that this line reports; the line is printed
# if one is newer than STAMP.
#
# Authors:       Claude Sonnet 5.5, Claude Opus 5.5
# Written:       13-October-2026
# Last update:   18-November-2026
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

set(changed OFF)
if(NOT EXISTS "${STAMP}")
  set(changed ON)
else()
  file(STRINGS "${COVERS_FILE}" covered)
  foreach(file IN LISTS covered)
    if("${file}" IS_NEWER_THAN "${STAMP}")
      set(changed ON)
      break()
    endif()
  endforeach()
endif()

# report once: the stamp is touched, so the line is printed only after the next rebuild
if(changed)
  if(BOLD)
    set(style --bold)
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green ${style} "${LABEL}")
  get_filename_component(folder "${STAMP}" DIRECTORY)
  file(MAKE_DIRECTORY "${folder}")
  file(TOUCH "${STAMP}")
endif()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
