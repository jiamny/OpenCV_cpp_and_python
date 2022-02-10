cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

include_directories( ${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------
add_executable(ColorizeImage ImageColorization.cpp)

target_link_libraries(ColorizeImage PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------
add_executable(ColorizeVideo VideoColorization.cpp)

target_link_libraries(ColorizeVideo PRIVATE ${OpenCV_LIBS})
