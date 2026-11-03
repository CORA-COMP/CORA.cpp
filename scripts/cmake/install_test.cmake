# install_test - installs the build into BUILD/install-test and builds and runs a tiny consumer
# against it (tests/install/consumer), with the flags of the build (a sanitized static library
# needs its runtime at the link); the ctest install/consumer.
#
#   cmake -DBUILD=build -DSOURCE=. -DGENERATOR=Ninja -DCXX=g++ -DCXX_FLAGS= -DCONFIG=Release -P install_test.cmake

set(prefix "${BUILD}/install-test/prefix")
set(consumer "${BUILD}/install-test/consumer")
file(REMOVE_RECURSE "${BUILD}/install-test")
if(NOT CONFIG)
  set(CONFIG Release)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD}" --prefix "${prefix}" --config "${CONFIG}"
                OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE}/tests/install/consumer" -B "${consumer}"
                        -G "${GENERATOR}" "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_CXX_FLAGS=${CXX_FLAGS}" "-DCMAKE_BUILD_TYPE=${CONFIG}"
                        "-DCMAKE_PREFIX_PATH=${prefix}"
                RESULT_VARIABLE configured)
if(NOT configured EQUAL 0)
  message(FATAL_ERROR "install/consumer: find_package(CORA) failed against ${prefix}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${consumer}" --config "${CONFIG}"
                COMMAND_ERROR_IS_FATAL ANY)
file(GLOB_RECURSE program "${consumer}/consumer" "${consumer}/consumer.exe")
list(GET program 0 program)
execute_process(COMMAND "${program}" RESULT_VARIABLE ran OUTPUT_VARIABLE out ERROR_VARIABLE out)
message("${out}")
if(NOT ran EQUAL 0)
  message(FATAL_ERROR "install/consumer: the program failed")
endif()
