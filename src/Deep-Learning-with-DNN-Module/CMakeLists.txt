
set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(DL_dnn_classify cpp/classify.cpp)

target_link_libraries(DL_dnn_classify PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(DL_dnn_detect_img cpp/detect_img.cpp)

target_link_libraries(DL_dnn_detect_img PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(DL_dnn_detect_vid cpp/detect_vid.cpp)

target_link_libraries(DL_dnn_detect_vid PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
