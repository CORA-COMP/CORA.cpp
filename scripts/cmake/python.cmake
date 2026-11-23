# python - the cora Python package: the compiled module and the Python files beside it
#
# Included by CMakeLists.txt; the variables and options are its.
#
# Authors:       Claude Sonnet 5.5
# Written:       30-September-2026
# Last update:   ---
# Reviewed:      ---

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ------------------------------------------- Python -----------------------------------------------
# The cora package: the compiled module and the Python files beside it, in <build>/cora. Put the
# build directory on PYTHONPATH. It runs on numpy and Eigen; with the torch package of the Python it
# also takes torch tensors (CORACPP_PYTHON_TORCH, the bindings' torch part). pybind11 comes with that
# torch, else from the Python environment, else it is fetched.

set(python_ok OFF)
set(python_missing "")
if(NOT CORACPP_PYTHON STREQUAL "OFF")
  if(NOT Python3_EXECUTABLE)
    set(python_missing "no Python 3 interpreter was found")
  elseif(NOT Python3_Development.Module_FOUND)
    # e.g. a system Python without its -dev package; a standalone Python (uv, conda) ships them
    string(CONCAT python_missing "${Python3_EXECUTABLE} has no development files (Python.h); install them "
                       "(e.g. python3-dev), or point -DPython3_EXECUTABLE=... at a Python that has them "
                       "(`uv python install 3.11`, conda)")
  else()
    # the package imports numpy, and the build imports the package to write its type stubs
    # the module carries the file name extension of the Python found
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
                    RESULT_VARIABLE numpy_result OUTPUT_QUIET ERROR_QUIET)
    if(numpy_result EQUAL 0)
      set(python_ok ON)
    else()
      set(python_missing "${Python3_EXECUTABLE} has no numpy")
    endif()
  endif()
  if(NOT python_ok AND CORACPP_PYTHON STREQUAL "AUTO")
    message(STATUS "Python: ${python_missing}, so the Python package is not built")
  endif()
endif()
if(CORACPP_PYTHON STREQUAL "ON" AND NOT python_ok)
  message(FATAL_ERROR "CORACPP_PYTHON=ON, but ${python_missing}")
endif()

if(python_ok)
  # the bindings take torch tensors when the torch of the Python environment is the libtorch in use
  set(python_torch OFF)
  if(CORACPP_HAS_TORCH)
    find_library(torch_python torch_python HINTS "${CORACPP_TORCH_LIB}" NO_DEFAULT_PATH)
    if(torch_python)
      set(python_torch ON)
    endif()
  endif()
  # pybind11 from the Python, else a fetched copy; with torch_python the bindings come from torch
  if(NOT python_torch)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_dir ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE pybind11_result)
    if(pybind11_result EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${pybind11_dir}")
    endif()
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
      include(FetchContent)
      FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
                           DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
      FetchContent_MakeAvailable(pybind11)
    endif()
  endif()

  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
                  "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
                  OUTPUT_VARIABLE extension_suffix OUTPUT_STRIP_TRAILING_WHITESPACE)
  # the module, linked to the library, and to torch_python or the headers of pybind11
  add_library(_cora MODULE "${CMAKE_SOURCE_DIR}/src/global/python/bindings.cpp")
  target_link_libraries(_cora PRIVATE cora Python3::Module)
  target_include_directories(_cora PRIVATE "${CMAKE_SOURCE_DIR}/src")
  if(python_torch)
    target_compile_definitions(_cora PRIVATE CORACPP_PYTHON_TORCH)
    target_link_directories(_cora PRIVATE "${CORACPP_TORCH_LIB}")
    target_link_libraries(_cora PRIVATE torch_python)
  else()
    target_link_libraries(_cora PRIVATE pybind11::headers)
  endif()
  # the module is built into build/cora, next to the Python files copied there
  set(module_directory "${CMAKE_BINARY_DIR}/cora")
  set_target_properties(_cora PROPERTIES PREFIX "" SUFFIX "${extension_suffix}" CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${module_directory}" RUNTIME_OUTPUT_DIRECTORY "${module_directory}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${module_directory}" LIBRARY_OUTPUT_DIRECTORY_RELEASE "${module_directory}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${module_directory}" RUNTIME_OUTPUT_DIRECTORY_RELEASE "${module_directory}")

  # the Python files of the package are copied beside the module when they change
  file(GLOB python_files CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/global/python/cora/*.py")
  set(copied)
  foreach(file IN LISTS python_files)
    get_filename_component(file_name "${file}" NAME)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/cora/${file_name}"
                       COMMAND "${CMAKE_COMMAND}" -E copy "${file}" "${CMAKE_BINARY_DIR}/cora/${file_name}"
                       DEPENDS "${file}")
    list(APPEND copied "${CMAKE_BINARY_DIR}/cora/${file_name}")
  endforeach()

  # the type stubs, written from the signatures of the module just built (scripts/stubs.py), and
  # py.typed (PEP 561) beside them: the editor completes and type-checks `cora`
  set(stub_files "${module_directory}/_cora/__init__.pyi" "${module_directory}/_cora/lean.pyi"
                 "${module_directory}/py.typed")
  add_custom_command(OUTPUT ${stub_files}
                     COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_BINARY_DIR}"
                             "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/stubs.py" --out "${module_directory}"
                     DEPENDS _cora ${copied} "${CMAKE_SOURCE_DIR}/scripts/stubs.py"
                     COMMENT "Generating cora/_cora/*.pyi")
  add_custom_target(python ALL DEPENDS _cora ${copied} ${stub_files})

  # the package, stubs included, for the install tree (put <prefix>/python on PYTHONPATH)
  install(DIRECTORY "${module_directory}/" DESTINATION python/cora PATTERN "__pycache__" EXCLUDE)

  cora_add_test(python python
                COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests/python
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  if(WIN32)
    set(python_path "${CMAKE_BINARY_DIR}")
  else()
    set(python_path "${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
  endif()
  # the examples the tests build and run use this build directory, in the current environment
  set_property(TEST python APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${python_path}"
               "CORACPP_BUILD=${CMAKE_BINARY_DIR}" "CORACPP_ENV=none")
  cora_run_environment(python)
endif()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
