cmake_minimum_required(VERSION 3.4.0)
project(educnn)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_OPENMP "Use OpenMP" ON)
if (WITH_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    message(STATUS "OpenMP found!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(sources)
