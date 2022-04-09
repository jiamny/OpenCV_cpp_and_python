


set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX__STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------------------
add_executable(Photoshop_filters-in_OpenCV_60s_TV  60s_TV.cpp)

target_link_libraries(Photoshop_filters-in_OpenCV_60s_TV PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Photoshop_filters-in_OpenCV_brightness  brightness.cpp)

target_link_libraries(Photoshop_filters-in_OpenCV_brightness PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Photoshop_filters-in_OpenCV_duo_tone  duo_tone.cpp)

target_link_libraries(Photoshop_filters-in_OpenCV_duo_tone PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Photoshop_filters-in_OpenCV_emboss  emboss.cpp)

target_link_libraries(Photoshop_filters-in_OpenCV_emboss PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# -------------------------------------------------------------------------------------
add_executable(Photoshop_filters-in_OpenCV_sepia  sepia.cpp)

target_link_libraries(Photoshop_filters-in_OpenCV_sepia PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

