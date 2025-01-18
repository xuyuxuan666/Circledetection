cmake_minimum_required(VERSION 3.10)

project(ImageProcessingProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5Widgets REQUIRED)

set(SOURCES
    main.cpp
    image_processing.cpp
    gridalignment.cpp
    sorting.cpp
    adjust.cpp
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageProcessing ${SOURCES})

target_link_libraries(ImageProcessing ${OpenCV_LIBS})
target_link_libraries(ImageProcessing Qt5::Widgets ${OpenCV_LIBS})
