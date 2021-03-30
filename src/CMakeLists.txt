execute_process(COMMAND bash setupvars.sh WORKING_DIRECTORY "~/intel/openvino_2021/bin/")

set(InferenceEngine_DIR "${CMAKE_SOURCE_DIR}/3rd/openvino/share/")

find_package(InferenceEngine REQUIRED CONFIG)
message("openvino lib have:" ${InferenceEngine_LIBRARIES})

add_executable(openvino_test kp2d_test.cpp KP2D.cpp)
#add_executable(openvino_test test.cpp)

set(OpenCV_DIR  "${CMAKE_SOURCE_DIR}/3rd/opencv_vino/cmake/")
find_package(OpenCV REQUIRED CONFIG)

set(TBB_DIR "${CMAKE_SOURCE_DIR}/3rd/openvino/external/tbb/cmake/")
find_package(TBB REQUIRED)
#file(GLOB TBB_LIBRARIES LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/3rd/openvino/external/tbb/lib/*.so")
#message("${TBB_LIBRARIES}")

#find_library(LIBINFERENCE_ENGINE_RUNTIME NAMES inference_engine_transformations inference_engine_legacy PATHS "${CMAKE_SOURCE_DIR}/3rd/openvino/lib/intel64/" )
#
#message("runtime lib is :${LIBINFERENCE_ENGINE_RUNTIME}")

set(ngraph_DIR "${CMAKE_SOURCE_DIR}/3rd/openvino/ngraph/cmake/")
find_package(ngraph REQUIRED)

message("opencv libs:" ${OpenCV_LIBS})
target_link_libraries(openvino_test ${OpenCV_LIBS} ${NGRAPH_LIBRARIES}  ${TBB_IMPORTED_TARGETS} ${InferenceEngine_LIBRARIES} )
target_include_directories(openvino_test PUBLIC ${OpenCV_INCLUDE_DIRS} ${InferenceEngine_INCLUDE_DIRS}  "${CMAKE_SOURCE_DIR}/include")
include_directories(${OpenCV_INCLUDE_DIRS} ${InferenceEngine_INCLUDE_DIRS}  "${CMAKE_SOURCE_DIR}/include")

