
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Otsu_implementation  otsu_implementation.cpp)

target_link_libraries(Otsu_implementation PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Otsu_method  otsu_method.cpp)

target_link_libraries(Otsu_method PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

