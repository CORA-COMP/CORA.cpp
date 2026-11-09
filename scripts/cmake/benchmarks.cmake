# benchmarks - the ARCH-COMP AFF benchmarks (examples/cpp/benchmarks), one program per family
# Included by CMakeLists.txt; the variables and functions are its.

# ------------------------------------------ benchmarks --------------------------------------------
# Each program reaches or verifies the instances of its family and prints the line of MATLAB CORA's
# benchmark_*.m; its exit code counts the verdicts that differ from MATLAB CORA's. Every instance is a
# test of the label `benchmark` (nightly only, generous timeout); one fast instance per family is
# also an `example` test, which the pull requests run.

file(GLOB benchmark_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/examples/cpp/benchmarks/benchmark_*.cpp")
foreach(source IN LISTS benchmark_sources)
  get_filename_component(name "${source}" NAME_WE)
  cora_executable(${name} "${source}" "benchmarks")
  target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/examples/cpp/benchmarks")
  target_compile_definitions(${name} PRIVATE
    "CORACPP_BENCHMARK_DATA=\"${CMAKE_SOURCE_DIR}/examples/benchmarks/data\"")
endforeach()

# the instances of each program, the first a smoke test of under a second (if the family has one)
set(benchmark_instances_benchmark_linear_verify_arch23_rand RAND01 RAND02)
set(benchmark_instances_benchmark_linear_verify_fast_arch23_beam CBC01 CBC02 CBC03 CBF01 CBF02 CBF03)
set(benchmark_instances_benchmark_linear_verify_fast_arch23_heat3d HEAT01 HEAT02)
set(benchmark_instances_benchmark_linear_verify_fast_arch23_iss
    ISSF01_ISU01 ISSC01_ISS02 ISSC01_ISU02 ISSF01_ISS01)
set(benchmark_instances_benchmark_hybrid_reach_arch23_rendezvous
    SRA01 SRA02 SRA03 SRA04 SRA05 SRA06 SRA07 SRA08 SRNA01 SRU01 SRU02)
set(benchmark_instances_benchmark_hybrid_reach_arch23_gearbox GRBX01 GRBX02)
set(benchmark_instances_benchmark_hybrid_reach_arch23_powertrain DTN01 DTN02 DTN03 DTN04 DTN05 DTN06)
set(benchmark_instances_benchmark_hybrid_reach_arch23_brake BRKDC01 BRKNC01)
set(benchmark_instances_benchmark_linear_param_reach_arch23_platoon
    PLAD01_BND42 PLAA01_BND42 PLAA01_BND50 PLAD01_BND30 PLAN01)
set(benchmark_instances_benchmark_linear_reach_arch23_heat3d HEAT03)
set(benchmark_instances_benchmark_nonlinear_reach_arch23_trafficscenario TRAF22)
set(benchmark_instances_benchmark_hybrid_reach_arch23_spacecraft SPRE22)
# the families whose fastest instance takes longer than a smoke test may (about 5 s)
# (the spacecraft: one smoke instance per change, TRAF22 is the nonlinear one)
set(benchmark_without_smoke_test benchmark_linear_reach_arch23_heat3d
    benchmark_hybrid_reach_arch23_spacecraft)
foreach(source IN LISTS benchmark_sources)
  get_filename_component(name "${source}" NAME_WE)
  set(first TRUE)
  if(name IN_LIST benchmark_without_smoke_test)
    set(first FALSE)
  endif()
  foreach(instance IN LISTS benchmark_instances_${name})
    cora_add_test(benchmark/${instance} benchmark COMMAND ${name} ${instance})
    set_tests_properties(benchmark/${instance} PROPERTIES TIMEOUT 3600)
    if(first)
      cora_add_test(examples/benchmark_${instance} example COMMAND ${name} ${instance})
      set(first FALSE)
    endif()
  endforeach()
endforeach()
