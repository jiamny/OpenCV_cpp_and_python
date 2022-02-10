cmake_minimum_required (VERSION 3.20)

set (CMAKE_CXX_STANDARD 11)

# ----------------------------------------------------------------------------------------
ADD_EXECUTABLE( BackgroundEstimation OpenCV_BackgroundEstimation.cpp )
TARGET_LINK_LIBRARIES( BackgroundEstimation ${OpenCV_LIBS} -lprotobuf ) 

