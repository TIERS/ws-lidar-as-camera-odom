cmake_minimum_required(VERSION 3.16.0)
project(superpoint_feature)

# find torch 
set(CMAKE_PREFIX_PATH "/home/jimmy/Downloads/Performance_comparison/ws-lidar-as-camera-odom/src/libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# find opencv
find_package(OpenCV 4.2.0 REQUIRED)

# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/incude
# )

message(">>>>>>>> SUB-SOURCE DIR: " ${CMAKE_CURRENT_SOURCE_DIR})

add_library(superpoint
        src/NetWork.cpp
        src/PointTracker.cpp
        src/SuperPointFrontend.cpp)

target_link_libraries(superpoint PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}) 

target_include_directories(superpoint PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include())

# # code to print include directories insize cmakelist.txt
# message("PROJECT INCLUDE DIRS: ")
# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message("    DIR=" ${dir})
# endforeach()
# add_executable(test test.cpp)

# target_link_libraries(test PUBLIC
#     superpoint
#     ${PROJECT_BINARY_DIR}
#     # ${TORCH_LIBRARIES}
#     # ${OpenCV_LIBS}
#     )
