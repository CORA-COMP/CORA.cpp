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
# Eigen (found, else fetched); the Python package needs libtorch and the Python development files.

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

option(CORACPP_NATIVE "Compile for this CPU (-march=native; not with MSVC)" ON)
option(CORACPP_BLAS "Let Eigen call an external BLAS (changes which library does the arithmetic)" OFF)
option(CORACPP_COMPETITION "Build the CORA-COMP binary and its tests (needs GLPK)" OFF)
set(CORACPP_TORCH AUTO CACHE STRING "libtorch: AUTO (if the Python has torch), ON, OFF")
set(CORACPP_PYTHON AUTO CACHE STRING "The Python package: AUTO (if torch and Python.h are there), ON, OFF")
set(CORACPP_TORCH_DIR "" CACHE PATH "The torch package directory (default: that of the Python found)")
set(CORACPP_TORCH_ABI "" CACHE STRING "The C++11 ABI of that torch, 0 or 1 (default: asked of the Python)")

# ------------------------------------------ dependencies -------------------------------------------

find_package(Eigen3 NO_MODULE)
if(NOT Eigen3_FOUND)
  # No installed Eigen (Windows, a bare macOS): take the release, header only.
  include(FetchContent)
  FetchContent_Declare(eigen3 URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
                       DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR headers-only)
  FetchContent_MakeAvailable(eigen3)
  # only the headers: Eigen's own CMake project would add its BLAS (Fortran) and tests
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
endif()
find_package(OpenMP COMPONENTS CXX)  # optional: Apple's clang has none without libomp
find_package(Python3 COMPONENTS Interpreter Development.Module)

add_library(cora_flags INTERFACE)
target_link_libraries(cora_flags INTERFACE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cora_flags INTERFACE OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP: not found, the library runs on one thread where it would use several")
endif()
if(MSVC)
  # /utf-8: the sources are UTF-8; /bigobj: Eigen templates; the defines keep <cmath> and <windows.h> tame.
  target_compile_options(cora_flags INTERFACE /utf-8 /bigobj /EHsc /W3
                         "$<$<CONFIG:Release>:/O2>")
  target_compile_definitions(cora_flags INTERFACE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS
                             "$<$<CONFIG:Release>:NDEBUG;EIGEN_NO_DEBUG>")
else()
  # Eigen's AVX512 triangular-solve kernel makes GCC 16 report uninitialized values and out-of-bounds
  # accesses that are not there; those two are off so that a real warning is visible.
  target_compile_options(cora_flags INTERFACE -Wall -Wextra -Wno-maybe-uninitialized -Wno-array-bounds)
  target_compile_options(cora_flags INTERFACE "$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Debug>:-O0;-g>")
  target_compile_definitions(cora_flags INTERFACE "$<$<CONFIG:Release>:NDEBUG;EIGEN_NO_DEBUG>")
  if(CORACPP_NATIVE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(cora_flags INTERFACE "$<$<CONFIG:Release>:-march=native>")
  endif()
endif()
if(MINGW)
  # The runtime DLLs of another MinGW on PATH (Git, Strawberry Perl, MATLAB) would be loaded first.
  target_link_options(cora_flags INTERFACE -static)
endif()
if(CORACPP_BLAS)
  target_compile_definitions(cora_flags INTERFACE EIGEN_USE_BLAS)
  target_link_libraries(cora_flags INTERFACE openblas)
endif()
if(UNIX AND DEFINED ENV{CONDA_PREFIX})
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
              "import os, torch; print(os.path.dirname(torch.__file__).replace(os.sep, '/'), int(torch._C._GLIBCXX_USE_CXX11_ABI))"
      OUTPUT_VARIABLE torch_info ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE torch_result)
    if(torch_result EQUAL 0)
      string(REGEX REPLACE " .*" "" torch_dir "${torch_info}")
      string(REGEX REPLACE ".* " "" torch_abi "${torch_info}")
    endif()
  endif()
  if(NOT CORACPP_TORCH_ABI STREQUAL "")
    set(torch_abi "${CORACPP_TORCH_ABI}")
  endif()
  # the import library (Windows) or shared object (Linux, macOS) that marks a usable torch package
  find_library(torch_core torch HINTS "${torch_dir}/lib" NO_DEFAULT_PATH)
  if(torch_dir AND torch_core)
    set(CORACPP_HAS_TORCH ON)
    target_compile_definitions(cora_torch INTERFACE CORACPP_TORCH)
    if(NOT WIN32 AND NOT APPLE)
      target_compile_definitions(cora_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=${torch_abi})
    endif()
    target_include_directories(cora_torch SYSTEM INTERFACE
      "${torch_dir}/include" "${torch_dir}/include/torch/csrc/api/include")
    target_link_directories(cora_torch INTERFACE "${torch_dir}/lib")
    target_link_libraries(cora_torch INTERFACE torch torch_cpu c10)
    # --no-as-needed: nothing references libtorch_cuda directly, but without it linked the CUDA
    # kernels are not registered and a GPU is never used.
    find_library(torch_cuda torch_cuda HINTS "${torch_dir}/lib" NO_DEFAULT_PATH)
    if(torch_cuda AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(cora_torch INTERFACE -Wl,--no-as-needed torch_cuda c10_cuda -Wl,--as-needed)
    elseif(torch_cuda)
      target_link_libraries(cora_torch INTERFACE torch_cuda c10_cuda)
    endif()
    set(CORACPP_TORCH_LIB "${torch_dir}/lib")
    if(UNIX)
      set(CMAKE_BUILD_RPATH "${CORACPP_TORCH_LIB}")  # the executables find libtorch without LD_LIBRARY_PATH
    endif()
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
# rng.cpp is the one file compiled with fast math, so the logarithm and the sine of Box-Muller
# vectorize; everything that results depend on keeps strict IEEE arithmetic.
if(MSVC)
  set(fast_math /fp:fast)
else()
  set(fast_math -ffast-math)
endif()
set_source_files_properties("${CMAKE_SOURCE_DIR}/src/global/rng.cpp" PROPERTIES COMPILE_OPTIONS "${fast_math}")

# ------------------------------------------- tests ------------------------------------------------
# tests/ mirrors src/; a test ending in _torch needs libtorch. test_codingConventions.py checks the
# conventions of the sources, and runs first.

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
function(cora_run_environment test)
  if(WIN32 AND CORACPP_HAS_TORCH)
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CORACPP_TORCH_LIB}")
  endif()
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
  cora_run_environment(${folder}/${name})
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
  if(MSVC)
    target_compile_options(${name} PRIVATE /FI global/banner.h)
  else()
    target_compile_options(${name} PRIVATE -include global/banner.h)
  endif()
  target_compile_definitions(${name} PRIVATE "CORACPP_PROGRAM=\"${name}.cpp\"")
  # the figures an example writes go to its own folder, not into the source tree
  add_test(NAME examples/${name} COMMAND ${name} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
  set_tests_properties(examples/${name} PROPERTIES LABELS example)
  cora_run_environment(examples/${name})
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
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
                  "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
                  OUTPUT_VARIABLE extension_suffix OUTPUT_STRIP_TRAILING_WHITESPACE)
  add_library(_cora MODULE "${CMAKE_SOURCE_DIR}/src/python/bindings.cpp")
  target_link_libraries(_cora PRIVATE cora Python3::Module)
  target_link_directories(_cora PRIVATE "${CORACPP_TORCH_LIB}")
  target_link_libraries(_cora PRIVATE torch_python)
  set(module_directory "${CMAKE_BINARY_DIR}/cora")
  set_target_properties(_cora PROPERTIES PREFIX "" SUFFIX "${extension_suffix}" CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${module_directory}" RUNTIME_OUTPUT_DIRECTORY "${module_directory}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${module_directory}" LIBRARY_OUTPUT_DIRECTORY_RELEASE "${module_directory}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${module_directory}" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${module_directory}")

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
  set_tests_properties(python PROPERTIES LABELS python)
  if(WIN32)
    set(python_path "${CMAKE_BINARY_DIR}")
  else()
    set(python_path "${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
  endif()
  set_property(TEST python APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${python_path}")
  cora_run_environment(python)
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
