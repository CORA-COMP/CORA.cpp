# python - the cora Python package: the compiled module and the Python files beside it
# Included by CMakeLists.txt; the variables and options are its.

# ------------------------------------------- Python -----------------------------------------------
# The cora package: the compiled module and the Python files beside it, in <build>/cora. Put the
# build directory on PYTHONPATH. It runs on numpy and Eigen; with the torch package of the Python it
# also takes torch tensors (CORACPP_PYTHON_TORCH, the bindings' torch part). pybind11 comes with that
# torch, else from the Python environment, else it is fetched.

set(python_ok OFF)
if(NOT CORACPP_PYTHON STREQUAL "OFF" AND Python3_Development.Module_FOUND)
  set(python_ok ON)
endif()
if(CORACPP_PYTHON STREQUAL "ON" AND NOT python_ok)
  message(FATAL_ERROR "CORACPP_PYTHON=ON needs Python with development files")
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
  add_library(_cora MODULE "${CMAKE_SOURCE_DIR}/src/python/bindings.cpp")
  target_link_libraries(_cora PRIVATE cora Python3::Module)
  if(python_torch)
    target_compile_definitions(_cora PRIVATE CORACPP_PYTHON_TORCH)
    target_link_directories(_cora PRIVATE "${CORACPP_TORCH_LIB}")
    target_link_libraries(_cora PRIVATE torch_python)
  else()
    target_link_libraries(_cora PRIVATE pybind11::headers)
  endif()
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
  # the examples the tests build and run use this build directory, in the current environment
  set_property(TEST python APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${python_path}"
               "CORACPP_BUILD=${CMAKE_BINARY_DIR}" "CORACPP_ENV=none")
  cora_run_environment(python)
endif()
