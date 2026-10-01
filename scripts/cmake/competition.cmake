# competition - the CORA-COMP harness (needs GLPK)
# Included by CMakeLists.txt; the variables and options are its.

# ---------------------------------------- the competition -----------------------------------------
# scripts/competition/ is the CORA-COMP harness and its sets; it builds on the library, needs GLPK.
# See scripts/competition/README.md.

if(CORACPP_COMPETITION)
  find_library(GLPK_LIBRARY glpk REQUIRED)
  file(GLOB_RECURSE comp_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/scripts/competition/*.cpp")
  list(FILTER comp_sources EXCLUDE REGEX "/scripts/competition/(main|torch_backend|torch_none)\\.cpp$")
  list(FILTER comp_sources EXCLUDE REGEX "/scripts/competition/sets/torch_contains\\.cpp$")
  if(CORACPP_HAS_TORCH)
    list(APPEND comp_sources "${CMAKE_SOURCE_DIR}/scripts/competition/torch_backend.cpp"
                             "${CMAKE_SOURCE_DIR}/scripts/competition/sets/torch_contains.cpp")
  else()
    list(APPEND comp_sources "${CMAKE_SOURCE_DIR}/scripts/competition/torch_none.cpp")
  endif()
  add_library(coracomp STATIC ${comp_sources})
  target_include_directories(coracomp PUBLIC "${CMAKE_SOURCE_DIR}/scripts/competition")
  target_link_libraries(coracomp PUBLIC cora "${GLPK_LIBRARY}")

  add_executable(coracpp "${CMAKE_SOURCE_DIR}/scripts/competition/main.cpp")
  target_link_libraries(coracpp PRIVATE coracomp)

  file(GLOB_RECURSE comp_tests CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/competition/test_*.cpp")
  foreach(source IN LISTS comp_tests)
    get_filename_component(name "${source}" NAME_WE)
    if(name MATCHES "_torch$" AND NOT CORACPP_HAS_TORCH)
      continue()
    endif()
    cora_executable(${name} "${source}" "tests/competition")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/tests")
    target_link_libraries(${name} PRIVATE coracomp)
    add_test(NAME competition/${name} COMMAND ${name})
    set_tests_properties(competition/${name} PROPERTIES LABELS competition)
  endforeach()
endif()
