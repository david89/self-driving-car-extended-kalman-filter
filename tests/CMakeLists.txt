project(ExtendedKFTests)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

add_library(tools ../src/tools.cpp)
add_executable(tools_test ../src/tools_test.cpp)
target_link_libraries(tools_test tools pthread gtest_main gtest)
