cmake_minimum_required(VERSION 3.10)
project(chipImg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED)

option(BUILD_C5 "Build C5 targets" ON)
option(BUILD_4X "Build 4X targets" ON)

set(PROJ_PUBLIC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

function(enable_warnings tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# ================== C5 版本 ==================
if(BUILD_C5)
  set(C5_SRC_DIR ${CMAKE_SOURCE_DIR}/src/C5)
  set(C5_MAIN    ${C5_SRC_DIR}/main_C5.cpp)

  if(NOT EXISTS ${C5_MAIN})
    message(FATAL_ERROR "C5: main file not found: ${C5_MAIN}")
  endif()

  add_library(cluster_c5 STATIC
    ${C5_SRC_DIR}/Cluster.cpp
    ${C5_SRC_DIR}/Anchor.cpp
    ${C5_SRC_DIR}/Grid.cpp
    ${C5_SRC_DIR}/MergeFilter.cpp
    ${C5_SRC_DIR}/OutputInterface_C5.cpp
    ${C5_SRC_DIR}/ShapeDetectionAPI_C5.cpp

  )
  target_include_directories(cluster_c5
    PUBLIC
      ${PROJ_PUBLIC_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(cluster_c5 PUBLIC ${OpenCV_LIBS})
  enable_warnings(cluster_c5)

  add_executable(C5 ${C5_MAIN})
  target_include_directories(C5 PRIVATE ${PROJ_PUBLIC_INCLUDE_DIR})
  target_link_libraries(C5 PRIVATE cluster_c5 ${OpenCV_LIBS})
  enable_warnings(C5)

  message(STATUS "C5 main: ${C5_MAIN}")
endif()

# ================== 4X 版本 ==================
if(BUILD_4X)
  add_library(cluster_4X STATIC
    src/4X/Cluster_4X.cpp
    src/4X/Anchor_4X.cpp
    src/4X/Grid_4X.cpp
    src/4X/MergeFilter_4X.cpp
    src/4X/OutputInterface_4X.cpp 
    src/4X/ShapeDetectionAPI_4X.cpp       
  )
  target_include_directories(cluster_4X
    PUBLIC
      ${PROJ_PUBLIC_INCLUDE_DIR}      
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(cluster_4X PUBLIC ${OpenCV_LIBS})
  enable_warnings(cluster_4X)

  add_executable(X4 src/4X/main_4X.cpp)
  target_link_libraries(X4 PRIVATE cluster_4X ${OpenCV_LIBS})
  enable_warnings(X4)
endif()



# ================== GMY 版本 ==================
option(BUILD_GMY "Build GMY targets" ON)

if(BUILD_GMY)
  add_library(cluster_GMY STATIC
    src/GMY/Cluster_GMY.cpp
    src/GMY/Anchor_GMY.cpp
    src/GMY/Grid_GMY.cpp
    src/GMY/MergeFilter_GMY.cpp
    src/GMY/OutputInterface_GMY.cpp
    src/GMY/ShapeDetectionAPI_GMY.cpp
  )
  target_include_directories(cluster_GMY
    PUBLIC
      ${PROJ_PUBLIC_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(cluster_GMY PUBLIC ${OpenCV_LIBS})
  enable_warnings(cluster_GMY)

  add_executable(GMY src/GMY/main_GMY.cpp)
  target_include_directories(GMY PRIVATE ${PROJ_PUBLIC_INCLUDE_DIR})
  target_link_libraries(GMY PRIVATE cluster_GMY ${OpenCV_LIBS})
  enable_warnings(GMY)

  message(STATUS "GMY enabled: builds libcluster_GMY + GMY executable")
endif()








# ================== PG 版本 ==================
option(BUILD_PG "Build PG targets" ON)

if(BUILD_PG)

  if(NOT BUILD_GMY)
    set(BUILD_GMY ON CACHE BOOL "Enable GMY because PG depends on it" FORCE)
  endif()

  add_library(cluster_PG STATIC
    src/PG/Cluster_PG.cpp
    src/PG/Anchor_PG.cpp
    src/PG/Grid_PG.cpp
    src/PG/MergeFilter_PG.cpp
    src/PG/OutputInterface_PG.cpp
    src/PG/ShapeDetectionAPI_PG.cpp
  )
  target_include_directories(cluster_PG
    PUBLIC
      ${PROJ_PUBLIC_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(cluster_PG PUBLIC ${OpenCV_LIBS} cluster_GMY)
  enable_warnings(cluster_PG)

  add_executable(PG src/PG/main_PG.cpp)
  target_include_directories(PG PRIVATE ${PROJ_PUBLIC_INCLUDE_DIR})
  target_link_libraries(PG PRIVATE cluster_PG ${OpenCV_LIBS})
  enable_warnings(PG)

  message(STATUS "PG enabled: builds libcluster_PG + PG executable (reusing GMY core)")

endif()


option(BUILD_STD "Build STD target (only main)" ON)

if(BUILD_STD)
  cmake_minimum_required(VERSION 3.14)

  project(STD_ONLY CXX)

  find_package(OpenCV REQUIRED)

  # 注意：这里定义的目标名是 std（小写），所以后面都要一致
  add_executable(std src/std/main_std.cpp)

  target_include_directories(std PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJ_PUBLIC_INCLUDE_DIR}  # 如果没有这个变量可以直接删掉
  )

  target_link_libraries(std PRIVATE ${OpenCV_LIBS})

  # 可选：开启常用编译告警
  if (MSVC)
    target_compile_options(std PRIVATE /W4 /permissive-)
  else()
    target_compile_options(std PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  message(STATUS "STD enabled: builds std executable (main only)")
endif()


