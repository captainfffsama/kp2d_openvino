cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(openvino_test LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src bin)
set_property(TARGET openvino_test PROPERTY CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Release config is:" ${CMAKE_CXX_FLAGS_RELEASE})


set(INSTALL_LIB_DIR "${CMAKE_SOURCE_DIR}/3rd/opencv_vino/lib/;${CMAKE_SOURCE_DIR}/3rd/openvino/ngraph/lib/;${CMAKE_SOURCE_DIR}/3rd/openvino/external/tbb/lib/;${CMAKE_SOURCE_DIR}/3rd/openvino/lib/intel64/")
set(CMAKE_SKIP_BUILD_RPATH FALSE)                 # 编译时加上RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)         # 编译时RPATH不使用安装的RPATH
set(CMAKE_INSTALL_RPATH ${INSTALL_LIB_DIR})
message(${CMAKE_INSTALL_RPATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
# 确保链接库不在系统默认安装的目录上时更改到项目lib上
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_RPATH}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${INSTALL_LIB_DIR}")
endif("${isSystemDir}" STREQUAL "-1")
