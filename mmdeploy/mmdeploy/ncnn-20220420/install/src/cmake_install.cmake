# Install script for directory: /home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/libncnn.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/allocator.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/benchmark.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/blob.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/c_api.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/command.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/cpu.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/datareader.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/gpu.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/layer.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/layer_shader_type.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/layer_type.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/mat.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/modelbin.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/net.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/option.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/paramdict.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/pipeline.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/pipelinecache.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/simpleocv.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/simpleomp.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/simplestl.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/src/vulkan_header_fix.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/ncnn_export.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/layer_shader_type_enum.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/layer_type_enum.h"
    "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/platform.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/ncnnConfig.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/dshengq/github/openmmlab/mmdeploy/mmdeploy/ncnn-20220420/install/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
