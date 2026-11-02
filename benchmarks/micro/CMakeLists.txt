# micro - the micro-benchmarks whose instruction counts CI compares between a base and a head
#
# A program outside CORA.cpp against an install tree, so the same source builds against any commit:
#   cmake -S benchmarks/micro -B build/micro -DCMAKE_PREFIX_PATH=<prefix of cmake --install>
# scripts/ci/instruction_counts.py runs it under valgrind.

cmake_minimum_required(VERSION 3.22)
project(micro LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CORA REQUIRED)
add_executable(micro micro.cpp)
target_link_libraries(micro PRIVATE CORA::cora)
