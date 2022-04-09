

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Stereo_camera_calibrate  calibrate.cpp)

target_link_libraries(Stereo_camera_calibrate PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Stereo_camera_capture_images  capture_images.cpp)

target_link_libraries(Stereo_camera_capture_images PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Stereo_camera_movie3d  movie3d.cpp)

target_link_libraries(Stereo_camera_movie3d PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
