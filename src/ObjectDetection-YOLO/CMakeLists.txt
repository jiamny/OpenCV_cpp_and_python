cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(ObjectDetectionYOLO_video  object_detection_video.cpp)

target_link_libraries(ObjectDetectionYOLO_video PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ObjectDetectionYOLO_img  yolov3_detect_img.cpp)

target_link_libraries(ObjectDetectionYOLO_img PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
