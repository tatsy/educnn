set(BUILD_TARGET "educnn")

set(SOURCES ${SOURCES}
    educnn.cpp
    activation.cpp
    mnist.cpp
    network.cpp
    fully_connected_layer.cpp
    convolution_layer.cpp
    max_pooling_layer.cpp
    average_pooling_layer.cpp)

set(HEADERS ${HEADERS}
    common.h
    activation.h
    mnist.h
    random.h
    network.h
    abstract_layer.h
    fully_connected_layer.h
    convolution_layer.h
    max_pooling_layer.h
    average_pooling_layer.h)

add_executable(${BUILD_TARGET} ${SOURCES} ${HEADERS})
source_group("Source Files" FILES ${SOURCES} ${HEADERS})

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
  set_property(TARGET ${BUILD_TARGET} APPEND PROPERTY LINK_FLAGS "/DEBUG /PROFILE")
endif()
