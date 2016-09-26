configure_file(directories.h.in
               ${CMAKE_CURRENT_LIST_DIR}/directories.h
               @ONLY)

set(EDUCNN_HEADERS ${EDUCNN_HEADERS}
    educnn.h
    common.h
    timer.h
    directories.h
    activation.h
    mnist.h
    random.h
    network.h
    abstract_layer.h
    fully_connected_layer.h
    convolution_layer.h
    max_pooling_layer.h
    average_pooling_layer.h)

function(BUILD_TEST_PROGRAM TEST_FILE BUILD_NAME)
  add_executable(${BUILD_NAME} ${TEST_FILE} ${EDUCNN_HEADERS})
  source_group("Source Files" FILES ${TEST_FILE} ${EDUCNN_HEADERS})

  if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
    set_property(TARGET ${BUILD_NAME} APPEND PROPERTY LINK_FLAGS "/DEBUG /PROFILE")
  endif()
endfunction()

build_test_program(test_neural.cpp test_neural)
build_test_program(test_lecun.cpp test_lecun)
