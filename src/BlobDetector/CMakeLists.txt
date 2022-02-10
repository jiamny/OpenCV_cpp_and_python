cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(BlobDetector blob.cpp)

target_link_libraries(BlobDetector PRIVATE ${OpenCV_LIBS})
