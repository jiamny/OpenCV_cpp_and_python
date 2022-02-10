cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(HandPoseImage  handPoseImage.cpp)

target_link_libraries(HandPoseImage PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(HandPoseVideo  handPoseVideo.cpp)

target_link_libraries(HandPoseVideo PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

