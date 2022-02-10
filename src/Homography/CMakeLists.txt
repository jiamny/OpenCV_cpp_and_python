cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Homography  homography.cpp)

target_link_libraries(Homography PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Homography2  homography2.cpp)

target_link_libraries(Homography2 PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Homography_perspective-correction  perspective-correction.cpp)

target_link_libraries(Homography_perspective-correction PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Homography_virtual_billboard  virtual-billboard.cpp)

target_link_libraries(Homography_virtual_billboard PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

