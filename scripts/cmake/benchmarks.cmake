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
set(benchmark_instances_benchmark_hybrid_reach_arch23_brake BRKDC01 BRKNC01 BRKNP01)
set(benchmark_instances_benchmark_linear_param_reach_arch23_platoon
    PLAD01_BND42 PLAA01_BND42 PLAA01_BND50 PLAD01_BND30 PLAN01)
set(benchmark_instances_benchmark_linear_reach_arch23_heat3d HEAT03)
set(benchmark_instances_benchmark_nonlinear_reach_arch23_trafficscenario TRAF22)
set(benchmark_instances_benchmark_hybrid_reach_arch23_spacecraft SPRE22)
set(benchmark_instances_benchmark_nonlinear_reach_arch23_robertson ROBE25_1 ROBE25_2 ROBE25_3)
set(benchmark_instances_benchmark_hybrid_reach_arch23_vanderpol CVDP23)
set(benchmark_instances_benchmark_hybrid_reach_arch23_lotkavolterra LOVO25 LOVO25-script)
set(benchmark_instances_benchmark_nonlinear_reach_arch23_laubloomis LALO20_W001 LALO20_W005 LALO20_W01)
set(benchmark_instances_benchmark_nonlinear_das_reach_arch23_powersystem TSPS25)
set(benchmark_instances_benchmark_neur_net_contr_reach_arch26_ainncs
    ACC-safe-distance Unicycle-reach Unicycle-reach-disturbance TORA-remain TORA-reach-tanh
    TORA-reach-sigmoid Single_Pendulum-reach Double_Pendulum-less-robust Double_Pendulum-more-robust
    Airplane-continuous Attitude_Control-avoid)
# the families whose fastest instance takes longer than a smoke test may (about 5 s)
# (the spacecraft: one smoke instance per change, TRAF22 is the nonlinear one; Robertson takes
# thousands of steps)
set(benchmark_without_smoke_test benchmark_linear_reach_arch23_heat3d
    benchmark_hybrid_reach_arch23_spacecraft benchmark_nonlinear_reach_arch23_robertson
    benchmark_hybrid_reach_arch23_vanderpol benchmark_nonlinear_reach_arch23_laubloomis
    benchmark_nonlinear_das_reach_arch23_powersystem
    benchmark_hybrid_reach_arch23_lotkavolterra)
foreach(source IN LISTS benchmark_sources)
  get_filename_component(name "${source}" NAME_WE)
  set(first TRUE)
  if(name IN_LIST benchmark_without_smoke_test)
    set(first FALSE)
  endif()
  foreach(instance IN LISTS benchmark_instances_${name})
    cora_add_test(benchmark/${instance} benchmark COMMAND ${name} ${instance})
    # exit 77: the instance keeps every set and needs more memory than the machine has (benchmark_run.h)
    set_tests_properties(benchmark/${instance} PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77)
    if(first)
      cora_add_test(examples/benchmark_${instance} example COMMAND ${name} ${instance})
      set(first FALSE)
    endif()
  endforeach()
endforeach()

# One verdict per family asked of the CORALean oracle at a sound dtype (benchmark_sound_verdict_arch23): a
# family the oracle cannot serve exits 77, which ctest reports as skipped. The label is `benchmark` (nightly).
foreach(family IN ITEMS linear hybrid nonlinear)
  cora_add_test(sound_verdict/${family} benchmark COMMAND benchmark_sound_verdict_arch23 ${family})
  set_tests_properties(sound_verdict/${family} PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77)
  cora_run_environment(sound_verdict/${family})
endforeach()
