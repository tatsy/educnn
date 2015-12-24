cmake_minimum_required(VERSION 2.8.0)
project(educnn)

find_package(OpenMP)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(sources)
