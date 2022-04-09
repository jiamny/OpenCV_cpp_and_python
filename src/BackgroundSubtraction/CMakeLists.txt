cmake_minimum_required (VERSION 3.18)

PROJECT(background_subtr)

set (CMAKE_CXX_STANDARD 11)


ADD_EXECUTABLE( BackgroundSubtraction_opencv background_subtr_opencv.cpp )
TARGET_LINK_LIBRARIES( BackgroundSubtraction_opencv ${OpenCV_LIBS} -lprotobuf ) 


