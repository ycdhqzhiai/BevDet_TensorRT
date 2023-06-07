cmake_minimum_required(VERSION 3.6)

project(bevdet)

SET(CMAKE_BUILD_TYPE "Debug")
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mavx -msse4.1 -mavx2 -D_GLIBCXX_USE_CXX11_ABI=1")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -g -Wfatal-errors -D_GLIBCXX_USE_CXX11_ABI=1")
option (USE_GPU "Use tutorial provided math implementation" ON)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(absl REQUIRED)


set(TRT_DIR $ENV{TENSORRT_DIR})

set(BEV_ENV_DIR $ENV{BEV_ENV})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${TRT_DIR}/include
    ${BEV_ENV_DIR}/include
    /usr/local/cuda-11.4/include
)
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

link_directories(
    ${BEV_ENV_DIR}/lib
    ${TRT_DIR}/lib
    /usr/local/cuda-11.4/lib64
)
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_DIRECTORIES)

file(GLOB_RECURSE SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    ./common/*.h
    ./common/*.cc
    ./base/*.h
    ./base/*.cc    
    ./proto/*.h
    ./proto/*.cc
    ./bevdet/*.cc
    ./bevdet/*.h
    ./bevdet/*.cu
    ./bevdet/*.h    
    ./src/*.cc
    ./src/*.h

)

cuda_add_executable(bevdet main.cc ${SRC_LIST})
target_link_libraries(bevdet ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} absl::strings  -ljsoncpp -lnppicc -lnppidei -lnvinfer -lcudart -lpthread -lprotobuf  -lglog gflags)

add_definitions(-O0 -pthread)
add_definitions(-DUSE_GPU)