cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(SeamlessCloning_clone  clone.cpp)

target_link_libraries(SeamlessCloning_clone PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(SeamlessCloning_normal_versus_mixed_clone  normal_versus_mixed_clone.cpp)

target_link_libraries(SeamlessCloning_normal_versus_mixed_clone PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(SeamlessCloning_laplacian_pyramid_blending  laplacian_pyramid_blending.cpp)

target_link_libraries(SeamlessCloning_laplacian_pyramid_blending PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
