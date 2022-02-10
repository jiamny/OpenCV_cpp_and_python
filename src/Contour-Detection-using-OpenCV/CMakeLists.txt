cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# ----------------------------------------------------------------
add_executable(Contour_Detection_channel_experiments cpp/channel_experiments.cpp)

target_link_libraries(Contour_Detection_channel_experiments PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# ----------------------------------------------------------------
add_executable(Contour_Detection_contour_approx   cpp/contour_approx.cpp)

target_link_libraries( Contour_Detection_contour_approx  PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# ----------------------------------------------------------------
add_executable(Contour_Detection_contour_extraction    cpp/contour_extraction.cpp)

target_link_libraries(Contour_Detection_contour_extraction  PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
