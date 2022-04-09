
set(CMAKE_CXX_STANDARD 14)

include_directories( ${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------
add_executable(ColorSpace_interactiveColorDetect interactiveColorDetect.cpp)

target_link_libraries(ColorSpace_interactiveColorDetect PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------
add_executable(ColorSpace_interactiveColorSegment interactiveColorSegment.cpp)

target_link_libraries(ColorSpace_interactiveColorSegment PRIVATE ${OpenCV_LIBS})
