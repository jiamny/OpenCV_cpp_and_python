
set(CMAKE_CXX_STANDARD 14)

include_directories( ${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------
add_executable(Colormap colormap.cpp)

target_link_libraries(Colormap PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------
add_executable(Custom_colormap custom_colormap.cpp)

target_link_libraries(Custom_colormap PRIVATE ${OpenCV_LIBS})
