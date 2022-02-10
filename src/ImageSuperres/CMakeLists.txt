cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(ImageSuperres_dnn_superres  dnn_superres.cpp)

target_link_libraries(ImageSuperres_dnn_superres PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ImageSuperres_dnn_superres_benchmark_quality  dnn_superres_benchmark_quality.cpp)

target_link_libraries(ImageSuperres_dnn_superres_benchmark_quality PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ImageSuperres_dnn_superres_benchmark_time  dnn_superres_benchmark_time.cpp)

target_link_libraries(ImageSuperres_dnn_superres_benchmark_time PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ImageSuperres_dnn_superres_multioutput  dnn_superres_multioutput.cpp)

target_link_libraries(ImageSuperres_dnn_superres_multioutput PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(ImageSuperres_dnn_superres_video  dnn_superres_video.cpp)

target_link_libraries(ImageSuperres_dnn_superres_video PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
