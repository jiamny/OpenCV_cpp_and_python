cmake_minimum_required(VERSION 3.20)


set(SOURCES
        src/main.cpp
        src/FaceDetector.cpp
        src/KeyPointDetector.cpp
        include/FaceDetector.h
        include/KeyPointDetector.h
        )

add_executable(Building_an_eye_tracker_with_openCV ${SOURCES})

target_link_libraries(Building_an_eye_tracker_with_openCV PRIVATE ${OpenCV_LIBS})

