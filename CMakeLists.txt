cmake_minimum_required(VERSION 3.16)
project(lab)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fopenmp -O3 -mfma")

add_executable(lab main.cpp)