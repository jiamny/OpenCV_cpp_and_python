cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Image_alignment  image_alignment.cpp)

target_link_libraries(Image_alignment  PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Image_alignment_simple_example  image_alignment_simple_example.cpp)

target_link_libraries(Image_alignment_simple_example  PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

