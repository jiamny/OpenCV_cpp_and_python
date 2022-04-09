

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Opencv_dlib_face_detection  face_detection.cpp dnn_opencv.cpp face_detection.h haar_opencv.cpp)

target_link_libraries(Opencv_dlib_face_detection PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

