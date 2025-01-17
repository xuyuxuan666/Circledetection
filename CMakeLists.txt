# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(ImageProcessingProject LANGUAGES CXX)

# 设置 C++ 标准为 C++11 或更高
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 OpenCV 库
find_package(OpenCV REQUIRED)
find_package(Qt5Widgets REQUIRED)

# 添加项目源文件
set(SOURCES
    main.cpp
    image_processing.cpp
    gridalignment.cpp
    sorting.cpp
    adjust.cpp
)

# 添加项目头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 定义可执行文件
add_executable(ImageProcessing ${SOURCES})

# 链接 OpenCV 库
target_link_libraries(ImageProcessing ${OpenCV_LIBS})
target_link_libraries(ImageProcessing Qt5::Widgets ${OpenCV_LIBS})
