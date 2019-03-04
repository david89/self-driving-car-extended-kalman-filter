project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/main.cpp src/tools.cpp src/FusionEkf.cpp src/kalman_filter.cpp src/tools.h src/FusionEkf.h src/kalman_filter.h)


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

  # For users with a custom homebrew installation.
  # TODO: Find a better way.
  if(EXISTS "$ENV{HOME}/homebrew")
    include_directories($ENV{HOME}/homebrew/opt/openssl/include)
    link_directories($ENV{HOME}/homebrew/opt/openssl/lib/)
    link_directories($ENV{HOME}/homebrew/Cellar/libuv/1*/lib)
  endif()

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 


add_executable(ExtendedKF ${sources})

target_link_libraries(ExtendedKF z ssl uv uWS)
