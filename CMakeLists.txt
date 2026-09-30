# CORA.cpp - the library, its tests and examples, the Python package, the competition binary
#
#   scripts/with_env.sh scripts/build.sh                 configure and build everything
#   scripts/with_env.sh ctest --test-dir build -j        conventions, C++ tests, examples, Python tests
#   scripts/build.sh --eigen                             without libtorch (Eigen only)
#   scripts/build.sh --debug <target>                    -O0 -g into build-debug
#   cmake -S . -B build && cmake --build build --target <name>     the same without the scripts
#
# Options (-D...): CORACPP_TORCH=AUTO|CORACPP_PYTHON=AUTO|ON|OFF (the Python package)|OFF (libtorch from the torch package of the Python found,
# or CORACPP_TORCH_DIR=/path), CORACPP_NATIVE=ON (-march=native), CORACPP_BLAS=OFF (external BLAS),
# CORACPP_PYTHON=AUTO|ON|OFF (the Python package), CORACPP_COMPETITION=OFF (needs GLPK).
# The library needs only Eigen and the compiler; the Python package needs libtorch and Python.h.

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

option(CORACPP_NATIVE "Compile for this CPU (-march=native)" ON)
option(CORACPP_BLAS "Let Eigen call an external BLAS (changes which library does the arithmetic)" OFF)
option(CORACPP_COMPETITION "Build the CORA-COMP binary and its tests (needs GLPK)" OFF)
set(CORACPP_TORCH AUTO CACHE STRING "libtorch: AUTO (if the Python has torch), ON, OFF")
set(CORACPP_PYTHON AUTO CACHE STRING "The Python package: AUTO (if torch and Python.h are there), ON, OFF")
set(CORACPP_TORCH_DIR "" CACHE PATH "The torch package directory (default: that of the Python found)")
set(CORACPP_TORCH_ABI "" CACHE STRING "The C++11 ABI of that torch, 0 or 1 (default: asked of the Python)")

# ------------------------------------------ dependencies -------------------------------------------

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python3 COMPONENTS Interpreter Development.Module)

# Eigen's AVX512 triangular-solve kernel makes GCC 16 report uninitialized values and out-of-bounds
# accesses that are not there; those two are off so that a real warning is visible.
add_library(cora_flags INTERFACE)
target_compile_options(cora_flags INTERFACE -Wall -Wextra -Wno-maybe-uninitialized -Wno-array-bounds)
target_compile_options(cora_flags INTERFACE "$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Debug>:-O0;-g>")
target_compile_definitions(cora_flags INTERFACE "$<$<CONFIG:Release>:NDEBUG;EIGEN_NO_DEBUG>")
if(CORACPP_NATIVE)
  target_compile_options(cora_flags INTERFACE "$<$<CONFIG:Release>:-march=native>")
endif()
target_link_libraries(cora_flags INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX)
if(CORACPP_BLAS)
  target_compile_definitions(cora_flags INTERFACE EIGEN_USE_BLAS)
  target_link_libraries(cora_flags INTERFACE openblas)
endif()
if(DEFINED ENV{CONDA_PREFIX})
  # The conda compiler's libstdc++ and the conda libraries are found at run time without an activation.
  target_include_directories(cora_flags SYSTEM INTERFACE "$ENV{CONDA_PREFIX}/include")
  target_link_options(cora_flags INTERFACE "-L$ENV{CONDA_PREFIX}/lib" "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
endif()

# libtorch: the directory of the torch package and the C++ ABI it was built with, asked of Python.
# The package's own CMake config is not used: it needs the CUDA toolkit when torch is a CUDA build.
add_library(cora_torch INTERFACE)
set(CORACPP_HAS_TORCH OFF)
if(NOT CORACPP_TORCH STREQUAL "OFF")
  set(torch_dir "${CORACPP_TORCH_DIR}")
  set(torch_abi 1)
  if(NOT torch_dir AND Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import os, torch; print(os.path.dirname(torch.__file__), int(torch._C._GLIBCXX_USE_CXX11_ABI))"
      OUTPUT_VARIABLE torch_info ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE torch_result)
    if(torch_result EQUAL 0)
      string(REGEX REPLACE " .*" "" torch_dir "${torch_info}")
      string(REGEX REPLACE ".* " "" torch_abi "${torch_info}")
    endif()
  endif()
  if(NOT CORACPP_TORCH_ABI STREQUAL "")
    set(torch_abi "${CORACPP_TORCH_ABI}")
  endif()
  if(torch_dir AND EXISTS "${torch_dir}/lib/libtorch.so")
    set(CORACPP_HAS_TORCH ON)
    target_compile_definitions(cora_torch INTERFACE CORACPP_TORCH _GLIBCXX_USE_CXX11_ABI=${torch_abi})
    target_include_directories(cora_torch SYSTEM INTERFACE
      "${torch_dir}/include" "${torch_dir}/include/torch/csrc/api/include")
    target_link_directories(cora_torch INTERFACE "${torch_dir}/lib")
    target_link_libraries(cora_torch INTERFACE torch torch_cpu c10)
    # --no-as-needed: nothing references libtorch_cuda directly, but without it linked the CUDA
    # kernels are not registered and a GPU is never used.
    if(EXISTS "${torch_dir}/lib/libtorch_cuda.so")
      target_link_libraries(cora_torch INTERFACE
        -Wl,--no-as-needed torch_cuda c10_cuda -Wl,--as-needed)
    endif()
    set(CORACPP_TORCH_LIB "${torch_dir}/lib")
    message(STATUS "libtorch: ${torch_dir} (C++11 ABI ${torch_abi})")
  elseif(CORACPP_TORCH STREQUAL "ON")
    message(FATAL_ERROR "CORACPP_TORCH=ON, but no libtorch: give CORACPP_TORCH_DIR or a Python with torch")
  else()
    message(STATUS "libtorch: not found, Eigen only")
  endif()
endif()

# ---------------------------------------- the library ---------------------------------------------
# src/ mirrors MATLAB CORA's folders, one operation per file; sources are found, so a new
# operation needs no edit here.

file(GLOB_RECURSE lib_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(FILTER lib_sources EXCLUDE REGEX "/src/python/")
list(FILTER lib_sources EXCLUDE REGEX "/src/tensor/torch\\.cpp$")
if(CORACPP_HAS_TORCH)
  list(APPEND lib_sources "${CMAKE_SOURCE_DIR}/src/tensor/torch.cpp")
endif()

add_library(cora STATIC ${lib_sources})
target_include_directories(cora PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(cora PUBLIC cora_flags)
if(CORACPP_HAS_TORCH)
  target_link_libraries(cora PUBLIC cora_torch)
endif()
# rng.cpp is the one file compiled with -ffast-math, so the logarithm and the sine of Box-Muller
# vectorize; everything that results depend on keeps strict IEEE arithmetic.
set_source_files_properties("${CMAKE_SOURCE_DIR}/src/global/rng.cpp" PROPERTIES COMPILE_OPTIONS -ffast-math)

# ------------------------------------------- tests ------------------------------------------------
# tests/ mirrors src/; a test ending in _torch needs libtorch. test_codingConventions.py checks the
# conventions of the sources, and runs first.

enable_testing()

# An executable `name` from `source`, in a folder of the build that mirrors the source tree.
function(cora_executable name source folder)
  add_executable(${name} "${source}")
  target_link_libraries(${name} PRIVATE cora)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${folder}")
endfunction()

if(Python3_EXECUTABLE)
  add_test(NAME conventions
           COMMAND "${Python3_EXECUTABLE}" tests/global/test_codingConventions.py
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

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
  target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/tests")
  add_test(NAME ${folder}/${name} COMMAND ${name})
  set_tests_properties(${folder}/${name} PROPERTIES LABELS test)
endforeach()

# ------------------------------------------ examples ----------------------------------------------
# One small file per topic in examples/cpp; the ones that name libtorch types need it. Every example
# starts with the CORA START block: it is injected here, not written in each file.

file(GLOB example_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/examples/cpp/example_*.cpp")
foreach(source IN LISTS example_sources)
  get_filename_component(name "${source}" NAME_WE)
  if(name MATCHES "(_gpu|_batch|_gradient)$" AND NOT CORACPP_HAS_TORCH)
    continue()
  endif()
  cora_executable(${name} "${source}" "examples/cpp")
  target_compile_options(${name} PRIVATE -include global/banner.h)
  target_compile_definitions(${name} PRIVATE "CORACPP_PROGRAM=\"${name}.cpp\"")
  # the figures an example writes go to its own folder, not into the source tree
  add_test(NAME examples/${name} COMMAND ${name} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
  set_tests_properties(examples/${name} PROPERTIES LABELS example)
endforeach()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

# ------------------------------------------- Python -----------------------------------------------
# The cora package: the compiled module and the Python files beside it, in <build>/cora. Put the
# build directory on PYTHONPATH. It needs libtorch, since torch tensors are its inputs.

# pybind11 comes with torch (its headers are in torch/include), so the module is a plain shared library.
set(python_ok OFF)
if(NOT CORACPP_PYTHON STREQUAL "OFF" AND CORACPP_HAS_TORCH AND Python3_Development.Module_FOUND)
  set(python_ok ON)
endif()
if(CORACPP_PYTHON STREQUAL "ON" AND NOT python_ok)
  message(FATAL_ERROR "CORACPP_PYTHON=ON needs libtorch and Python with development files")
endif()

if(python_ok)
  add_library(_cora MODULE "${CMAKE_SOURCE_DIR}/src/python/bindings.cpp")
  target_link_libraries(_cora PRIVATE cora Python3::Module)
  set_target_properties(_cora PROPERTIES PREFIX "" SUFFIX ".${Python3_SOABI}.so" CXX_VISIBILITY_PRESET hidden)
  target_link_directories(_cora PRIVATE "${CORACPP_TORCH_LIB}")
  target_link_libraries(_cora PRIVATE torch_python)
  set_target_properties(_cora PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cora")

  file(GLOB python_files CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/python/cora/*.py")
  set(copied)
  foreach(file IN LISTS python_files)
    get_filename_component(file_name "${file}" NAME)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/cora/${file_name}"
                       COMMAND "${CMAKE_COMMAND}" -E copy "${file}" "${CMAKE_BINARY_DIR}/cora/${file_name}"
                       DEPENDS "${file}")
    list(APPEND copied "${CMAKE_BINARY_DIR}/cora/${file_name}")
  endforeach()
  add_custom_target(python ALL DEPENDS _cora ${copied})

  add_test(NAME python
           COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests/python
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(python PROPERTIES LABELS python
                       ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
endif()

# ---------------------------------------- the competition -----------------------------------------
# competition/ is the CORA-COMP harness and its own sets; it builds on the library and needs GLPK.
# See competition/README.md.

if(CORACPP_COMPETITION)
  find_library(GLPK_LIBRARY glpk REQUIRED)
  file(GLOB_RECURSE comp_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/competition/*.cpp")
  list(FILTER comp_sources EXCLUDE REGEX "/competition/(main|torch_backend|torch_none)\\.cpp$")
  list(FILTER comp_sources EXCLUDE REGEX "/competition/sets/torch_contains\\.cpp$")
  if(CORACPP_HAS_TORCH)
    list(APPEND comp_sources "${CMAKE_SOURCE_DIR}/competition/torch_backend.cpp"
                             "${CMAKE_SOURCE_DIR}/competition/sets/torch_contains.cpp")
  else()
    list(APPEND comp_sources "${CMAKE_SOURCE_DIR}/competition/torch_none.cpp")
  endif()
  add_library(coracomp STATIC ${comp_sources})
  target_include_directories(coracomp PUBLIC "${CMAKE_SOURCE_DIR}/competition")
  target_link_libraries(coracomp PUBLIC cora "${GLPK_LIBRARY}")

  add_executable(coracpp "${CMAKE_SOURCE_DIR}/competition/main.cpp")
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
