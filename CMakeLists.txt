cmake_minimum_required(VERSION 3.0.2)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(AeonGUI CXX)
find_package(OpenGL)

if(WIN32)
  find_program(ASTYLE_EXECUTABLE astyle
               HINTS ENV ProgramData
               PATH_SUFFIXES /usr/bin chocolatey/bin)
  find_program(FIND_EXECUTABLE find
               HINTS ENV ProgramFiles
               PATH_SUFFIXES /usr/bin Git/usr/bin)
else()
  find_program(FIND_EXECUTABLE find)
  find_program(ASTYLE_EXECUTABLE astyle)
endif()

find_program(XXD_EXECUTABLE xxd HINTS ENV ProgramData PATH_SUFFIXES /usr/bin chocolatey/bin)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_ZLIB "Use the zlib compression library")
option(USE_PNG "Enable PNG image support (Requires zlib)")
option(USE_CUDA "Enable CUDA support")
option(BUILD_UNIT_TESTS "Enable Unit Tests using GTest/GMock")
option(RELEASE_BUILD "Build for Release")
set(HTTP_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")
set(HTTPS_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")

if(NOT "${HTTP_PROXY}" STREQUAL "")
	set(ENV{http_proxy} "${HTTP_PROXY}")
endif()
if(NOT "${HTTPS_PROXY}" STREQUAL "")
	set(ENV{https_proxy} "${HTTPS_PROXY}")
endif()

if(USE_CUDA)
	message(STATUS "Looking for CUDA toolkit")
	find_package(CUDA)
	add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

if(USE_ZLIB)
	find_package(zlib)
endif(USE_ZLIB)
if(USE_PNG)
	find_package(libpng)
endif(USE_PNG)

include(opensans)
find_package(FreeType)
if(FREETYPE_FOUND)
	include_directories(${FREETYPE_INCLUDE_DIRS})
endif()
find_package(LibXml2)
if(LIBXML2_FOUND)
	include_directories(${LIBXML2_INCLUDE_DIR})
endif()

include(FindX11)
if(CMAKE_COMPILER_IS_GNUCXX)
	# -std=c++0x is required to be able to include stdint.h as cstdint, weird, I know.
	# But then CUDA doesn't like c++11, so stick to .h files until it does *sigh*
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	if(RELEASE_BUILD)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -fomit-frame-pointer -O3")
	else(RELEASE_BUILD)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
	endif(RELEASE_BUILD)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_COMPILER_IS_GNUC)
	set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -fomit-frame-pointer -O3")
	set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -g -O0")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif(CMAKE_COMPILER_IS_GNUC)

if(BUILD_UNIT_TESTS)
	include(gmock)
endif()

include_directories(include common common/pcx demos/common core)
subdirs(common/pcx core tools demos)
