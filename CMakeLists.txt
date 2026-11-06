# CORA.cpp - the library, its tests and examples, the Python package, the competition binary
#
#   cmake -S . -B build && cmake --build build --parallel       configure and build everything
#   ctest --test-dir build -j4 --output-on-failure              conventions, C++ tests, examples, Python tests
#   cmake -S . -B build -DCORACPP_TORCH=OFF                     without libtorch (Eigen only)
#   cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug          -O0 -g
#   scripts/with_env.sh scripts/build.sh                        the same inside the conda environment (Linux, WSL2)
#
# Linux, macOS and Windows (MSVC, MinGW, clang). Options (-D...): CORACPP_TORCH=AUTO|ON|OFF
# (libtorch from the torch package of the Python found, or CORACPP_TORCH_DIR=/path),
# CORACPP_NATIVE=ON (-march=native), CORACPP_BLAS=OFF (external BLAS), CORACPP_PYTHON=AUTO|ON|OFF
# (the Python package), CORACPP_COMPETITION=OFF (needs GLPK, Linux). The library needs the compiler and
# Eigen (found, else fetched); the Python package needs the Python development files and numpy.

cmake_minimum_required(VERSION 3.22)

# An activated conda environment sets CXXFLAGS to its own -O2 flags; the build type decides here.
unset(ENV{CXXFLAGS})
project(CORA LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release (-O3), Debug (-O0 -g)" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the Python module is a shared object
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CORACPP_NATIVE "Compile for this CPU (-march=native; /arch:AVX2 with MSVC)" ON)
option(CORACPP_WERROR "Warnings are errors (the presets and CI set it; -DCORACPP_WERROR=OFF opts out locally)" OFF)
option(CORACPP_BLAS "Let Eigen call an external BLAS (changes which library does the arithmetic)" OFF)
option(CORACPP_COMPETITION "Build the CORA-COMP binary and its tests (needs GLPK)" OFF)
option(CORACPP_CHECKS "Compile the input checks of the public entry points in (OFF: compiled out)" ON)
set(CORACPP_TORCH AUTO CACHE STRING "libtorch: AUTO (if the Python has torch), ON, OFF, DOWNLOAD (fetch a CPU libtorch)")
set(CORACPP_TORCH_VERSION "2.14.0" CACHE STRING "The libtorch that CORACPP_TORCH=DOWNLOAD fetches")
set(CORACPP_TORCH_VARIANT "cpu" CACHE STRING "cpu, or a CUDA variant such as cu126 (Linux and Windows)")
option(CORACPP_VENV "Create .venv and pip install CORACPP_VENV_REQUIREMENTS into it, and use its Python" OFF)
set(CORACPP_VENV_REQUIREMENTS "scripts/requirements-python-torch.txt" CACHE STRING "The requirements file the virtual environment gets")
set(CORACPP_VENV_TORCH_INDEX "https://download.pytorch.org/whl/cpu" CACHE STRING "Where pip gets torch for the virtual environment (.../whl/cu126 for CUDA)")
set(CORACPP_PYTHON AUTO CACHE STRING "The Python package: AUTO (if Python.h and numpy are there), ON, OFF")
set(CORACPP_TORCH_DIR "" CACHE PATH "The torch package directory (default: that of the Python found)")
set(CORACPP_TORCH_ABI "" CACHE STRING "The C++11 ABI of that torch, 0 or 1 (default: asked of the Python)")

include(scripts/cmake/dependencies.cmake)
include(scripts/cmake/oracle.cmake)

include(scripts/cmake/torch.cmake)

# ---------------------------------------- the library ---------------------------------------------
# include/cora/ holds the public headers, src/ the implementation and the private headers; both
# mirror MATLAB CORA's folders, one operation per file. Sources are found, so a new operation needs
# no edit here. cora/config.h records the options that change the headers.

if(CORACPP_CHECKS)
  set(CORACPP_CHECKS_VALUE 1)
else()
  set(CORACPP_CHECKS_VALUE 0)
endif()
configure_file("${CMAKE_SOURCE_DIR}/include/cora/config.h.in" "${CMAKE_BINARY_DIR}/include/cora/config.h" @ONLY)

file(GLOB_RECURSE lib_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(FILTER lib_sources EXCLUDE REGEX "/src/global/python/")
list(FILTER lib_sources EXCLUDE REGEX "/src/global/backend/torch\\.cpp$")
if(CORACPP_HAS_TORCH)
  list(APPEND lib_sources "${CMAKE_SOURCE_DIR}/src/global/backend/torch.cpp")
endif()

include(scripts/cmake/library.cmake)
# rng.cpp is the one file compiled with fast math, so the logarithm and the sine of Box-Muller
# vectorize; everything that results depend on keeps strict IEEE arithmetic.
if(MSVC)
  set(fast_math /fp:fast)
else()
  set(fast_math -ffast-math)
endif()
set_source_files_properties("${CMAKE_SOURCE_DIR}/src/global/rng.cpp" PROPERTIES COMPILE_OPTIONS "${fast_math}")
# The lean backend (the reference operations of host_tensor.cpp) rounds every product and every sum,
# as CORALean does: no fused multiply-add.
if(NOT MSVC)
  set_source_files_properties("${CMAKE_SOURCE_DIR}/src/global/backend/host_tensor.cpp" PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# ------------------------------------------- tests ------------------------------------------------
# tests/ mirrors src/; a test ending in _torch needs libtorch. test_codingConventions.py checks the
# conventions of the sources, and coverage.py --check (the tags, the coverage tables, the ratchet) run first.

enable_testing()

# An executable `name` from `source`, in a folder of the build that mirrors the source tree (for every
# configuration of a multi-config generator too).
function(cora_executable name source folder)
  add_executable(${name} "${source}")
  target_link_libraries(${name} PRIVATE cora)
  set(directory "${CMAKE_BINARY_DIR}/${folder}")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${directory}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${directory}" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${directory}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${directory}")
endfunction()

# Windows finds libtorch's DLLs through PATH: the tests that run a program get it in front.
# ctest runs the tests side by side, so each gets one thread (a test about threads sets its own).
function(cora_run_environment test)
  set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT "CORACPP_NUM_THREADS=1")
  if(WIN32 AND CORACPP_HAS_TORCH)
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CORACPP_TORCH_LIB}")
  endif()
  # the tests that talk to CORALean find the oracle through these variables
  if(coracpp_oracle_env)
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT "${coracpp_oracle_env}")
  endif()
endfunction()

if(Python3_EXECUTABLE)
  add_test(NAME coverage
           COMMAND "${Python3_EXECUTABLE}" scripts/coverage/coverage.py --check
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME coverage/checker
           COMMAND "${Python3_EXECUTABLE}" tests/global/test_coverage.py
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME conventions
           COMMAND "${Python3_EXECUTABLE}" tests/global/test_coding_conventions.py
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# the tests of the scripts (scripts/ci): tests/global/test_*.py, but the conventions test above
file(GLOB_RECURSE script_tests CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/global/test_*.py")
foreach(source IN LISTS script_tests)
  get_filename_component(name "${source}" NAME_WE)
  if(Python3_EXECUTABLE AND NOT name STREQUAL "test_coding_conventions")
    add_test(NAME global/${name} COMMAND "${Python3_EXECUTABLE}" "${source}")
    set_tests_properties(global/${name} PROPERTIES LABELS test)
  endif()
endforeach()

file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/test_*.cpp")
foreach(source IN LISTS test_sources)
  file(RELATIVE_PATH relative "${CMAKE_SOURCE_DIR}/tests" "${source}")
  if(relative MATCHES "^competition/")
    continue()
  endif()
  get_filename_component(name "${source}" NAME_WE)
  get_filename_component(folder "${relative}" DIRECTORY)
  if(name MATCHES "_torch$" AND NOT CORACPP_HAS_TORCH)
    continue()
  endif()
  cora_executable(${name} "${source}" "tests/${folder}")
  target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/tests" "${CMAKE_SOURCE_DIR}/src")
  add_test(NAME ${folder}/${name} COMMAND ${name})
  set_tests_properties(${folder}/${name} PROPERTIES LABELS test)
  cora_run_environment(${folder}/${name})
endforeach()

# ------------------------------------------ examples ----------------------------------------------
# One small file per topic in examples/cpp; the ones that name libtorch types need it. Every example
# starts with the CORA START block: it is injected here, not written in each file.

file(GLOB example_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/examples/cpp/example_*.cpp")
foreach(source IN LISTS example_sources)
  get_filename_component(name "${source}" NAME_WE)
  # an example that includes the torch header needs libtorch
  file(READ "${source}" example_text)
  if(example_text MATCHES "backend/torch\\.h" AND NOT CORACPP_HAS_TORCH)
    continue()
  endif()
  cora_executable(${name} "${source}" "examples/cpp")
  if(MSVC)
    target_compile_options(${name} PRIVATE /FI cora/global/banner.h)
  else()
    target_compile_options(${name} PRIVATE -include cora/global/banner.h)
  endif()
  target_compile_definitions(${name} PRIVATE "CORACPP_PROGRAM=\"${name}.cpp\"")
  # the figures an example writes go to its own folder, not into the source tree
  add_test(NAME examples/${name} COMMAND ${name} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
  set_tests_properties(examples/${name} PROPERTIES LABELS example)
  cora_run_environment(examples/${name})
endforeach()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

include(scripts/cmake/install.cmake)

include(scripts/cmake/python.cmake)

include(scripts/cmake/benchmarks.cmake)

include(scripts/cmake/competition.cmake)
