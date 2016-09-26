cmake_minimum_required(VERSION 2.8.0)
project(educnn)

set(CMAKE_CXX_STANDARD 11)
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
endif()

find_package(OpenMP)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(sources)
