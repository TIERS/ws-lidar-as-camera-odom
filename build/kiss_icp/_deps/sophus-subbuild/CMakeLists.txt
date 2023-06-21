# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(sophus-populate NONE)

include(ExternalProject)
ExternalProject_Add(sophus-populate
                     "UPDATE_DISCONNECTED" "False" "SYSTEM" "URL" "https://github.com/nachovizzo/Sophus/archive/refs/tags/1.22.11.tar.gz"
                    SOURCE_DIR          "/home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp/_deps/sophus-src"
                    BINARY_DIR          "/home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp/_deps/sophus-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
