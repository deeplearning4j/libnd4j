cmake_minimum_required(VERSION 3.2)
project(libnd4j)
set( CMAKE_VERBOSE_MAKEFILE on )
option(TESTS "Compile and make tests for the code?" OFF)
option(PTX "Compile ptx" OFF)
option(CUBIN "Compile cubin" OFF)
option(ECLIPSE "Generate eclipse files" OFF)
option(BLAS "Compile blas shared library for either cuda or cpu" ON)
option(NATIVE "Optimize for build machine (might not work on others)" OFF)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

find_path(STDIO_INCLUDE_PATH stdio.h)

set(CMAKE_CXX_FLAGS_RELEASE  "-Wall -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0 -std=c++11")



if(NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()



add_subdirectory(include)

find_package(CUDA)

include_directories(include/ include/helpers/ include/loops/ include/ops/ include/types/)
add_executable(main main.cpp include/helpers/files.h tests/ShapeTests.cpp include/helpers/data_gen.h)

target_link_libraries(main gtest gtest_main)




if(ECLIPSE)
    message("ADDING ECLIPSE")
    add_subdirectory(eclipse)
endif()

if(PTX)
    message("ADDING PTX")
    add_subdirectory(ptx)
endif()

if(CUBIN)
    message("ADDING CUBIN")
    add_subdirectory(cubin)
endif()


if(BLAS)
    message("ADDING BLAS")
    add_subdirectory(blas)
endif()

if(FATBIN)
    message("ADDING FATBIN")
    add_subdirectory(fatbin)
endif()

if(TESTS)
    # packages
    message("ADDING TEST")
    add_subdirectory(test)
endif()

if( PACKAGING STREQUAL "deb" )
    message("ADDING DEB")
    add_subdirectory(deb)
endif()

if( PACKAGING STREQUAL "rpm" )
    message("ADDING RPM")
    add_subdirectory(rpm)
endif()
