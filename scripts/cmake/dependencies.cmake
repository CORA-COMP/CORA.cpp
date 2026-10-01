# dependencies - Eigen, the virtual environment, the Python found and the compiler flags of cora_flags
# Included by CMakeLists.txt; the variables and options are its.

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
if(CORACPP_VENV)
  # The virtual environment of the project: created once, filled from the requirements, and the
  # Python that the rest of the configuration uses.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(venv "${CMAKE_SOURCE_DIR}/.venv")
  if(WIN32)
    set(venv_python "${venv}/Scripts/python.exe")
  else()
    set(venv_python "${venv}/bin/python")
  endif()
  if(NOT EXISTS "${venv_python}")
    message(STATUS "creating ${venv}")
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${venv}" COMMAND_ERROR_IS_FATAL ANY)
  endif()
  message(STATUS "installing ${CORACPP_VENV_REQUIREMENTS} into ${venv}")
  execute_process(COMMAND "${venv_python}" -m pip install -q -r "${CMAKE_SOURCE_DIR}/${CORACPP_VENV_REQUIREMENTS}"
                          --extra-index-url "${CORACPP_VENV_TORCH_INDEX}"
                  COMMAND_ERROR_IS_FATAL ANY)
  unset(Python3_EXECUTABLE CACHE)
  set(Python3_EXECUTABLE "${venv_python}")
  set(Python3_FIND_VIRTUALENV ONLY)
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module)

add_library(cora_flags INTERFACE)
target_link_libraries(cora_flags INTERFACE Eigen3::Eigen)
if(MINGW)
  # MinGW's libgomp needs libgomp-1.dll next to the module, and its static form needs dlerror.
  message(STATUS "OpenMP: not used with MinGW, the library runs on one thread")
elseif(OpenMP_CXX_FOUND)
  target_link_libraries(cora_flags INTERFACE OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP: not found, the library runs on one thread where it would use several")
endif()
if(MSVC)
  # /utf-8: the sources are UTF-8; /bigobj: Eigen templates; the defines keep <cmath> and <windows.h> tame.
  target_compile_options(cora_flags INTERFACE /utf-8 /bigobj /EHsc /W3
                         "$<$<CONFIG:Release>:/O2>")
  # MSVC has no -march=native; AVX2 is what the x64 CPUs of the last decade run.
  if(CORACPP_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    target_compile_options(cora_flags INTERFACE "$<$<CONFIG:Release>:/arch:AVX2>")
  endif()
  target_compile_definitions(cora_flags INTERFACE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS
                             "$<$<CONFIG:Release>:NDEBUG;EIGEN_NO_DEBUG>")
else()
  # Eigen's AVX512 triangular-solve kernel makes GCC 16 report uninitialized values and out-of-bounds
  # accesses that are not there; those two are off so that a real warning is visible.
  target_compile_options(cora_flags INTERFACE -Wall -Wextra)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cora_flags INTERFACE -Wno-maybe-uninitialized -Wno-array-bounds)
  endif()
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
