cmake_minimum_required(VERSION 3.20)

project(barcode_scaner)

find_package(Threads REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(ZBAR REQUIRED zbar)

add_executable(Barcode_scaner barcode-QRcodeScanner.cpp)
target_include_directories(Barcode_scaner PRIVATE ${OpenCV_INCLUDE_DIRS} ${ZBAR_INCLUDE_DIRS})
target_link_directories(Barcode_scaner PRIVATE ${ZBAR_LIBRARY_DIRS})
target_link_libraries(Barcode_scaner PRIVATE ${OpenCV_LIBS} ${ZBAR_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
