# consumer - a program outside CORA.cpp that finds the installed package (the ctest install/consumer)

cmake_minimum_required(VERSION 3.22)
project(consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CORA REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE CORA::cora)
