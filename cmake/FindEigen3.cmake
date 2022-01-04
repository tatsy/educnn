include(FindPackageHandleStandardArgs)

set(EIGEN3_DIR CACHE PATH "")

find_path(EIGEN3_INCLUDE_DIR
          NAMES Eigen/Core
          PATHS ${EIGEN3_DIR})

find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

if (EIGEN3_FOUND)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
  message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
endif()

mark_as_advanced(${EIGEN3_INCLUDE_DIR})
