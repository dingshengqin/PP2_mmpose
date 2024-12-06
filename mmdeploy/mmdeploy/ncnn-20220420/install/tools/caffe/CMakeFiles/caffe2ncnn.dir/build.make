# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install

# Include any dependencies generated for this target.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/compiler_depend.make

# Include the progress variables for this target.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make

tools/caffe/caffe.pb.h: /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe.proto
tools/caffe/caffe.pb.h: /home/dshengq/anaconda3/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running cpp protocol buffer compiler on caffe.proto"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /home/dshengq/anaconda3/bin/protoc --cpp_out :/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe -I /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe.proto

tools/caffe/caffe.pb.cc: tools/caffe/caffe.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate tools/caffe/caffe.pb.cc

tools/caffe/CMakeFiles/caffe2ncnn.dir/codegen:
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/codegen

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe2ncnn.cpp
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o -MF CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.d -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o -c /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe2ncnn.cpp

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe2ncnn.cpp > CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe/caffe2ncnn.cpp -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe/caffe.pb.cc
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o -MF CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.d -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o -c /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe/caffe.pb.cc

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe/caffe.pb.cc > CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe/caffe.pb.cc -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s

# Object files for target caffe2ncnn
caffe2ncnn_OBJECTS = \
"CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o" \
"CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"

# External object files for target caffe2ncnn
caffe2ncnn_EXTERNAL_OBJECTS =

tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make
tools/caffe/caffe2ncnn: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable caffe2ncnn"
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/caffe/CMakeFiles/caffe2ncnn.dir/build: tools/caffe/caffe2ncnn
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/build

tools/caffe/CMakeFiles/caffe2ncnn.dir/clean:
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe && $(CMAKE_COMMAND) -P CMakeFiles/caffe2ncnn.dir/cmake_clean.cmake
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/clean

tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe/caffe.pb.cc
tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe/caffe.pb.h
	cd /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420 /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/tools/caffe /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/tools/caffe/CMakeFiles/caffe2ncnn.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/depend

