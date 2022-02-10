cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(forEach  forEach.cpp)

target_link_libraries(forEach PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

