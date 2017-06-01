include(FindPackageHandleStandardArgs)

find_path(EIGEN3_INCLUDE_DIR
          NAMES Eigen/Core
          PATHS ${EIGEN3_ROOT})

find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

if (EIGEN3_FOUND)
  set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
  message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
  set(EIGEN3_ROOT "EIGEN3_ROOT" CACHE PATH "")
  message(FATAL_ERROR "Eigen3 not found! Please specify \"EIGEN3_ROOT\"")
endif()

mark_as_advanced(${EIGEN3_INCLUDE_DIR})
