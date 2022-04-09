
set(CMAKE_CXX_STANDARD 14)

include_directories( ${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------
add_executable(CenterofBlob_center_of_multipe_blob center_of_multipe_blob.cpp)

target_link_libraries(CenterofBlob_center_of_multipe_blob PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------
add_executable(CenterofBlob_single_blob single_blob.cpp)

target_link_libraries(CenterofBlob_single_blob PRIVATE ${OpenCV_LIBS})
