
PROJECT(AgeGender)

set (CMAKE_CXX_STANDARD 11)

ADD_EXECUTABLE( AlphaBlend alphaBlend.cpp )
TARGET_LINK_LIBRARIES( AlphaBlend ${OpenCV_LIBS} -lprotobuf ) 

