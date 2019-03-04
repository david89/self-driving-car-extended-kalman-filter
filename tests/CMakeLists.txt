project(ExtendedKFTests)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

add_library(tools ../src/tools.cpp)
add_executable(tools_test ../src/tools_test.cpp)
target_link_libraries(tools_test tools pthread gtest_main gtest)

add_library(kalman_filter ../src/kalman_filter.cpp)
add_executable(kalman_filter_test ../src/kalman_filter_test.cpp)
target_link_libraries(kalman_filter_test kalman_filter pthread gtest_main gtest)
