
set (CMAKE_CXX_STANDARD 11)

# ----------------------------------------------------------------------------------------
ADD_EXECUTABLE( AugmentedRealityWithArucoMarkers_aruco augmented_reality_with_aruco.cpp )
TARGET_LINK_LIBRARIES( AugmentedRealityWithArucoMarkers_aruco ${OpenCV_LIBS} -lprotobuf ) 

# ----------------------------------------------------------------------------------------
ADD_EXECUTABLE( AugmentedRealityWithArucoMarkers_markers generate_aruco_markers.cpp )
TARGET_LINK_LIBRARIES( AugmentedRealityWithArucoMarkers_markers ${OpenCV_LIBS} -lprotobuf ) 


