# progress_stamp - prints a build report line when something it covers was rebuilt
#
# Run by the custom targets of scripts/cmake/library.cmake: cmake -DCOVERS=<a|b|...> -DSTAMP=<file>
# -DLABEL=<text> [-DBOLD=ON] -P progress_stamp.cmake. COVERS are the object files, or the stamps of
# the folders below, that this line reports; the line is printed if one is newer than STAMP.

set(changed OFF)
if(NOT EXISTS "${STAMP}")
  set(changed ON)
else()
  string(REPLACE "|" ";" covered "${COVERS}")
  foreach(file IN LISTS covered)
    if("${file}" IS_NEWER_THAN "${STAMP}")
      set(changed ON)
      break()
    endif()
  endforeach()
endif()

if(changed)
  if(BOLD)
    set(style --bold)
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green ${style} "${LABEL}")
  get_filename_component(folder "${STAMP}" DIRECTORY)
  file(MAKE_DIRECTORY "${folder}")
  file(TOUCH "${STAMP}")
endif()
