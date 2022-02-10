cmake_minimum_required (VERSION 3.20)

PROJECT(background_subtr)

set (CMAKE_CXX_STANDARD 11)


# Requires OpenCV
#FIND_PACKAGE( OpenCV 4.5.3 REQUIRED )
#MESSAGE("OpenCV version : ${OpenCV_VERSION}")


ADD_EXECUTABLE( BackgroundSubtraction_opencv background_subtr_opencv.cpp )
TARGET_LINK_LIBRARIES( BackgroundSubtraction_opencv ${OpenCV_LIBS} -lprotobuf ) 


