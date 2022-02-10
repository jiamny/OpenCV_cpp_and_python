cmake_minimum_required (VERSION 3.0)

PROJECT(AgeGender)

set (CMAKE_CXX_STANDARD 11)


# Requires OpenCV
#FIND_PACKAGE( OpenCV 4.5.3 REQUIRED )
#MESSAGE("OpenCV version : ${OpenCV_VERSION}")


ADD_EXECUTABLE( AlphaBlend alphaBlend.cpp )
TARGET_LINK_LIBRARIES( AlphaBlend ${OpenCV_LIBS} -lprotobuf ) 

