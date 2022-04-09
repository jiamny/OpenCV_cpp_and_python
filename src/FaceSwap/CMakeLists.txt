

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(FaceSwap faceSwap.cpp)

target_link_libraries(FaceSwap PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

