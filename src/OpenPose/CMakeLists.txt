

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(OpenPose_image  OpenPoseImage.cpp)

target_link_libraries(OpenPose_image PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(OpenPose_video  OpenPoseVideo.cpp)

target_link_libraries(OpenPose_video PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
