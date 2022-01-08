configure_file(directories.h.in
               ${CMAKE_CURRENT_LIST_DIR}/directories.h
               @ONLY)

set(EDUCNN_SOURCES
    main.cpp
    educnn.h
    common.h
    timer.h
    progress.h
    directories.h
    openmp.h
    activation.h
    losses.h
    mnist.h
    random.h
    network.h
    abstract_layer.h
    fully_connected_layer.h
    convolution_layer.h
    max_pooling_layer.h
    average_pooling_layer.h)

add_executable(educnn ${EDUCNN_SOURCES})
source_group("Source Files" FILES ${EDUCNN_SOURCES})
set_target_properties(educnn PROPERTIES DEBUG_POSTFIX "-debug")

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
  set_property(TARGET ${BUILD_NAME} APPEND PROPERTY LINK_FLAGS "/DEBUG /PROFILE /INCREMENTAL:NO")
endif()
