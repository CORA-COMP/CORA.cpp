# torch - libtorch from the torch package of the Python found (target cora_torch)
#
# Included by CMakeLists.txt; the variables and options are its.
#
# Authors:       Claude Sonnet 5.5
# Written:       30-September-2026
# Last update:   ---
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# libtorch: the directory of the torch package and the C++ ABI it was built with, asked of Python.
# The package's own CMake config is not used: it needs the CUDA toolkit when torch is a CUDA build.
add_library(cora_torch INTERFACE)
set(CORACPP_HAS_TORCH OFF)
if(NOT CORACPP_TORCH STREQUAL "OFF")
  set(torch_dir "${CORACPP_TORCH_DIR}")
  set(torch_abi 1)
  if(CORACPP_TORCH STREQUAL "DOWNLOAD" AND NOT torch_dir)
    # the release of libtorch from pytorch.org, as the build of this machine: header and binaries only
    set(torch_root "https://download.pytorch.org/libtorch/${CORACPP_TORCH_VARIANT}")
    if(WIN32)
      set(torch_url "${torch_root}/libtorch-win-shared-with-deps-${CORACPP_TORCH_VERSION}%2B${CORACPP_TORCH_VARIANT}.zip")
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND CORACPP_TORCH_VARIANT STREQUAL "cpu")
      set(torch_url "${torch_root}/libtorch-macos-arm64-${CORACPP_TORCH_VERSION}.zip")
    elseif(APPLE)
      message(FATAL_ERROR "CORACPP_TORCH=DOWNLOAD: pytorch.org offers libtorch for macOS on Apple silicon only")
    else()
      set(torch_url "${torch_root}/libtorch-shared-with-deps-${CORACPP_TORCH_VERSION}%2B${CORACPP_TORCH_VARIANT}.zip")
    endif()
    # the archive is fetched here, not by FetchContent, so that a failure can name the other way
    get_filename_component(torch_archive "${torch_url}" NAME)
    set(torch_archive "${CMAKE_BINARY_DIR}/_deps/${torch_archive}")
    # a .part file until it is complete, so that an interrupted download is not taken for one
    if(NOT EXISTS "${torch_archive}")
      message(STATUS "fetching libtorch: ${torch_url}")
      file(DOWNLOAD "${torch_url}" "${torch_archive}.part" STATUS torch_status)
      list(GET torch_status 0 torch_code)
      if(NOT torch_code EQUAL 0)
        list(GET torch_status 1 torch_error)
        file(REMOVE "${torch_archive}.part")
        message(FATAL_ERROR "CORACPP_TORCH=DOWNLOAD: ${torch_url} failed (${torch_error}). Where "
                            "pytorch.org/libtorch is not reachable, the torch wheel is: use the "
                            "python-torch or python-torch-cuda preset, or pip install torch and give "
                            "-DCORACPP_TORCH=ON -DPython3_EXECUTABLE=<that Python> (see README.md)")
      endif()
      file(RENAME "${torch_archive}.part" "${torch_archive}")
    endif()
    include(FetchContent)
    FetchContent_Declare(libtorch URL "${torch_archive}" SOURCE_SUBDIR headers-only DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(libtorch)
    set(torch_dir "${libtorch_SOURCE_DIR}")
  endif()
  # without a download the package of the Python found gives the directory and the C++11 ABI
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
      # MSVC drops an unreferenced DLL import library: this symbol keeps torch_cuda loaded.
      if(MSVC)
        target_link_options(cora_torch INTERFACE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
      endif()
    endif()
    set(CORACPP_TORCH_LIB "${torch_dir}/lib")
    if(UNIX)
      set(CMAKE_BUILD_RPATH "${CORACPP_TORCH_LIB}")  # the executables find libtorch without LD_LIBRARY_PATH
    endif()
    message(STATUS "libtorch: ${torch_dir} (C++11 ABI ${torch_abi})")
  elseif(CORACPP_TORCH STREQUAL "ON" OR CORACPP_TORCH STREQUAL "DOWNLOAD")
    message(FATAL_ERROR "CORACPP_TORCH=${CORACPP_TORCH}, but no libtorch: give CORACPP_TORCH_DIR, a Python with torch, or use DOWNLOAD")
  else()
    message(STATUS "libtorch: not found, Eigen only")
  endif()
endif()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
