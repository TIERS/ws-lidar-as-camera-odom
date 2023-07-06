cmake_minimum_required(VERSION 3.26.0)
project(superpoint_feature)


# find torch 
set(CMAKE_PREFIX_PATH "./libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
# find opencv
find_package(OpenCV 4.2.0 REQUIRED)

message("SUB-SOURCE DIR: " ${CMAKE_CURRENT_SOURCE_DIR})

add_library(superpoint
        src/NetWork.cpp
        src/PointTracker.cpp
        src/SuperPointFrontend.cpp)

target_link_libraries(superpoint PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}) 

target_include_directories(superpoint PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


# add_executable(test test.cpp)

# target_link_libraries(test PUBLIC
#     superpoint
#     ${PROJECT_BINARY_DIR}
#     # ${TORCH_LIBRARIES}
#     # ${OpenCV_LIBS}
#     )
