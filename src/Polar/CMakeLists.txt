

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Polar  OpenCV_Polar.cpp)

target_link_libraries(Polar PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Polar_wrap  OpenCV_WarpPolar.cpp)

target_link_libraries(Polar_wrap PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
