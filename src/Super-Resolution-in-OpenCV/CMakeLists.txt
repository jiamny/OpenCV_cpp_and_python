
project(SuperResolution)

if(MSVC)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Super-Resolution-in-OpenCV super_res.cpp)

target_link_libraries(Super-Resolution-in-OpenCV ${OpenCV_LIBS})

set_property(TARGET Super-Resolution-in-OpenCV PROPERTY CXX_STANDARD 14)
