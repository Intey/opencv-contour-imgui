cmake_minimum_required(VERSION 3.0.2)
project(segmentator)

get_filename_component(THIS_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

cmake_policy(SET CMP0072 NEW)
# for VIM YCM Language engine
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/3rdparty/imgui)

find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
# find_package(PkgConfig REQUIRED)
# pkg_search_module(glfw REQUIRED glfw3)

find_package(OpenGL REQUIRED)

file(GLOB SOURCES src/*.cpp)

add_executable(${THIS_DIR_NAME} 
  # ${IMGUI_DIR}/examples/example_glfw_opengl3/main.cpp
  ${SOURCES}
  ${IMGUI_DIR}/imgui.cpp 
  ${IMGUI_DIR}/imgui_draw.cpp 
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
  )

target_include_directories(${THIS_DIR_NAME}
  PRIVATE 
  ${OpenCV_INCLUDE_DIRS}
  ${IMGUI_DIR} 
  ${IMGUI_DIR}/backends
  ${GLFW_INCLUDE_DIRS}
  )


target_link_libraries(${THIS_DIR_NAME} 
  PUBLIC
  ${OpenCV_LIBRARIES}
  OpenGL::OpenGL
  glfw
  # ${GLFW_LIBRARIES}
  ${CMAKE_DL_LIBS}
  )

set_target_properties(${THIS_DIR_NAME}
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17)

get_target_property(INCLUDE_DIRS_FOR_YCM ${THIS_DIR_NAME} INCLUDE_DIRECTORIES)

foreach(dir ${INCLUDE_DIRS_FOR_YCM})
  string(APPEND INCLUDE_COMPILER_STRING "-I${dir} ")
endforeach()

file(WRITE "${CMAKE_BINARY_DIR}/${THIS_DIR_NAME}.includes.list" ${INCLUDE_COMPILER_STRING})

