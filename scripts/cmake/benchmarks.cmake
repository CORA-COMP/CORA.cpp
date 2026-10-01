# benchmarks - the ARCH-COMP AFF benchmarks (examples/cpp/benchmarks), one program per family
# Included by CMakeLists.txt; the variables and functions are its.

# ------------------------------------------ benchmarks --------------------------------------------
# Each program verifies the instances of examples/benchmarks/data with LinearSys::verify and prints the
# line of MATLAB CORA's benchmark_*.m. They are not tests; run build/benchmarks/<name>.
# They build once LinearSys declares verify.

file(READ "${CMAKE_SOURCE_DIR}/src/contDynamics/linearSys/linearSys.h" linear_sys_header)
if(linear_sys_header MATCHES "VerifyParams")
  file(GLOB benchmark_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/examples/cpp/benchmarks/benchmark_*.cpp")
  foreach(source IN LISTS benchmark_sources)
    get_filename_component(name "${source}" NAME_WE)
    cora_executable(${name} "${source}" "benchmarks")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/examples/cpp/benchmarks")
    target_compile_definitions(${name} PRIVATE
      "CORACPP_BENCHMARK_DATA=\"${CMAKE_SOURCE_DIR}/examples/benchmarks/data\"")
  endforeach()
endif()
