
set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(ECCAlignment OpenCV_ECCAlignment.cpp)

target_link_libraries(ECCAlignment PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ECCAlignment_example OpenCV_ECCAlignment_example.cpp)

target_link_libraries(ECCAlignment_example PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

