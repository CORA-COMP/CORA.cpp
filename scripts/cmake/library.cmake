# library - the target cora, built per folder of src/ so that the build reports each folder when done
# Included by CMakeLists.txt; lib_sources is its.

# A folder of src/ is a group: src/<top>/<class>/ for a class folder, src/<top>/ otherwise. Each
# group is an object library; once its objects are built, a stamp prints "ok <group>", and the top
# folders that have class folders print a summary after their classes.
if(WIN32)
  set(done_mark "ok")  # the Windows console does not show UTF-8 by default
else()
  set(done_mark "✔")
endif()

set(group_names "")
foreach(source IN LISTS lib_sources)
  file(RELATIVE_PATH relative "${CMAKE_SOURCE_DIR}/src" "${source}")
  string(REPLACE "/" ";" parts "${relative}")
  list(LENGTH parts count)
  list(GET parts 0 top)
  set(leaf "${top}")
  if(count GREATER 2)
    list(GET parts 1 second)
    if(NOT second STREQUAL "private")
      set(leaf "${top}/${second}")
    endif()
  endif()
  string(MAKE_C_IDENTIFIER "${leaf}" id)
  list(APPEND group_${id}_sources "${source}")
  if(NOT leaf IN_LIST group_names)
    list(APPEND group_names "${leaf}")
    set(group_${id}_leaf "${leaf}")
    set(group_${id}_top "${top}")
  endif()
endforeach()

# the top folders that have class folders
set(top_names "")
foreach(leaf IN LISTS group_names)
  string(MAKE_C_IDENTIFIER "${leaf}" id)
  if(NOT leaf STREQUAL group_${id}_top)
    list(APPEND top_names "${group_${id}_top}")
  endif()
endforeach()
list(REMOVE_DUPLICATES top_names)

# the stamps depend on each other in this order, so that the report reads as a tree whatever order
# the objects finish in (the objects themselves do not wait for it)
list(SORT group_names)
set(tops "")
foreach(leaf IN LISTS group_names)
  string(MAKE_C_IDENTIFIER "${leaf}" id)
  list(APPEND tops "${group_${id}_top}")
endforeach()
list(REMOVE_DUPLICATES tops)

set(object_expressions "")
set(previous "")
foreach(top IN LISTS tops)
  foreach(leaf IN LISTS group_names)
    string(MAKE_C_IDENTIFIER "${leaf}" id)
    if(NOT group_${id}_top STREQUAL top)
      continue()
    endif()
    add_library(cora_${id} OBJECT ${group_${id}_sources})
    target_include_directories(cora_${id} PUBLIC "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(cora_${id} PUBLIC cora_flags)
    if(CORACPP_HAS_TORCH)
      target_link_libraries(cora_${id} PUBLIC cora_torch)
    endif()
    list(APPEND object_expressions "$<TARGET_OBJECTS:cora_${id}>")

    # a class folder is indented under its top folder, whose summary follows
    if(NOT top IN_LIST top_names)
      set(label "${done_mark} ${leaf}/")
    elseif(leaf STREQUAL top)
      set(label "  ${done_mark} ${leaf}/ (files)")
    else()
      set(label "  ${done_mark} ${leaf}/")
    endif()
    set(stamp "${CMAKE_BINARY_DIR}/progress/${id}.stamp")
    add_custom_command(OUTPUT "${stamp}"
      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "${label}"
      COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
      DEPENDS $<TARGET_OBJECTS:cora_${id}> ${previous} VERBATIM)
    set(previous "${stamp}")
  endforeach()
  if(top IN_LIST top_names)
    set(stamp "${CMAKE_BINARY_DIR}/progress/top_${top}.stamp")
    add_custom_command(OUTPUT "${stamp}"
      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green --bold "${done_mark} ${top}/"
      COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
      DEPENDS ${previous} VERBATIM)
    set(previous "${stamp}")
  endif()
endforeach()
add_custom_target(cora_progress DEPENDS ${previous})

add_library(cora STATIC ${object_expressions})
add_dependencies(cora cora_progress)
target_include_directories(cora PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(cora PUBLIC cora_flags)
if(CORACPP_HAS_TORCH)
  target_link_libraries(cora PUBLIC cora_torch)
endif()
