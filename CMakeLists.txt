cmake_minimum_required(VERSION 3.16)
project(iARAP)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

# Libigl
option(LIBIGL_GLFW         "Build target igl::glfw"         ON)
option(LIBIGL_IMGUI        "Build target igl::imgui"        ON)
option(LIBIGL_OPENGL       "Build target igl::opengl"       ON)
option(LIBIGL_PNG          "Build target igl::png"          ON)
include(libigl)

add_subdirectory("geometry-central")

# Add your project files
file(GLOB SRC_FILES src/*.cpp)
file(GLOB HEADER_FILES include/*.hpp)

file(GLOB VIEWER_FILES viewer/*.cpp)

add_library(iarap_core STATIC ${SRC_FILES} ${HEADER_FILES})
target_include_directories(iarap_core PUBLIC include)
target_link_libraries(iarap_core PUBLIC geometry-central igl::core)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC iarap_core)

add_executable(${PROJECT_NAME}_gui ${VIEWER_FILES})
target_link_libraries(${PROJECT_NAME}_gui PUBLIC iarap_core igl::glfw igl::opengl igl::imgui igl::png)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
