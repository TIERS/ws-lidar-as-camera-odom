# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jimmy/ws-lidar-as-camera-odom/src/kiss-icp/ros

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp

# Utility rule file for tests.

# Include the progress variables for this target.
include CMakeFiles/tests.dir/progress.make

tests: CMakeFiles/tests.dir/build.make

.PHONY : tests

# Rule to build all files generated by this target.
CMakeFiles/tests.dir/build: tests

.PHONY : CMakeFiles/tests.dir/build

CMakeFiles/tests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tests.dir/clean

CMakeFiles/tests.dir/depend:
	cd /home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jimmy/ws-lidar-as-camera-odom/src/kiss-icp/ros /home/jimmy/ws-lidar-as-camera-odom/src/kiss-icp/ros /home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp /home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp /home/jimmy/ws-lidar-as-camera-odom/build/kiss_icp/CMakeFiles/tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tests.dir/depend

