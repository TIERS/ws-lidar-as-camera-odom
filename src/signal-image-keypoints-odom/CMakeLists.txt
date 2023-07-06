cmake_minimum_required(VERSION 3.26.0)
project(signal-image-keypoints-odom)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  roscpp
  rospy
  std_msgs
  tf
)

# find opencv
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# find torch 
set(CMAKE_PREFIX_PATH "./libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# include pcl libraries
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(PCL REQUIRED COMPONENTS common io)

# add_subdirectory(src/superpoint/)

###################################
## catkin specific configuration ##
###################################

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES target-surface-detection
#  CATKIN_DEPENDS cv_bridge geometry_msgs image_transport roscpp rospy std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

message("PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR})
# message("PROJECT_BINARY_DIR:" ${${PROJECT_BINARY_DIR}})

# code to print include directories insize cmakelist.txt
message("PROJECT INCLUDE DIRS: ")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message("    DIR=" ${dir})
endforeach()

## Declare a C++ library
add_library(superpoint
  src/superpoint/NetWork.cpp
  src/superpoint/PointTracker.cpp
  src/superpoint/SuperPointFrontend.cpp
)

# target_include_directories(signal_image_keypoints_odom
#     PUBLIC 
#         ${PROJECT_SOURCE_DIR}/include
# )

target_link_libraries(superpoint PUBLIC
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

add_library(signal_image_keypoints_odom
  src/signal_image_keypoints_odom.cpp
)


target_link_libraries(signal_image_keypoints_odom PUBLIC
  superpoint
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(signal_image_keypoints_odom_node src/signal_image_keypoints_odom_node.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(signal_image_keypoints_odom_node PUBLIC
  signal_image_keypoints_odom
  ${catkin_LIBRARIES}
)

# # list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/superpoint")
# target_include_directories(signal_image_keypoints_odom_node PUBLIC
#   # ${PROJECT_BINARY_DIR}
#   ${PROJECT_SOURCE_DIR}/include
#   # ${EXTRA_INCLUDES}
# )

