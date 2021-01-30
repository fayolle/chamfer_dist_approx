cmake_minimum_required(VERSION 3.1)
project(chamfer_dist)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${PROJECT_SOURCE_DIR}/eigen)

add_executable(chamfer_dist src/main.cpp)
