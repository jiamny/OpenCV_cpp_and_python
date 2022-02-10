cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

include_directories( ${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------
add_executable(CameraCalibration cameraCalibration.cpp)

target_link_libraries(CameraCalibration PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------
add_executable(CameraCalibrationWithUndistortion cameraCalibrationWithUndistortion.cpp)

target_link_libraries(CameraCalibrationWithUndistortion PRIVATE ${OpenCV_LIBS})
