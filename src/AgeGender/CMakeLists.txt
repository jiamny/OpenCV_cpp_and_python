
PROJECT(AgeGender)

set (CMAKE_CXX_STANDARD 11)

# Requires OpenCV

ADD_EXECUTABLE( AgeGender AgeGender.cpp )
TARGET_LINK_LIBRARIES( AgeGender ${OpenCV_LIBS} -lprotobuf ) 

