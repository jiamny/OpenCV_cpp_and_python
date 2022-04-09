

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(OpenPose-Multi-Person  multi-person-openpose.cpp)

target_link_libraries(OpenPose-Multi-Person PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

