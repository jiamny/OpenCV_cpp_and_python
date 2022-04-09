

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(VideoReadWriteDisplay_read  videoRead.cpp)

target_link_libraries(VideoReadWriteDisplay_read PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(VideoReadWriteDisplay_write  videoWrite.cpp)

target_link_libraries(VideoReadWriteDisplay_write PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
