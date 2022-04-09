

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Object_tracking_use_OpenCV  Object_tracking.cpp)

target_link_libraries(Object_tracking_use_OpenCV PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

