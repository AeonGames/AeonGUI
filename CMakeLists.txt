cmake_minimum_required(VERSION 3.2)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(AeonGUI CXX)
find_package(OpenGL)
include(astyle)

#option(USE_FREETYPE  "Use FreeType2 for font rendering")
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

if(WIN32)
	include(xxd)
endif()

if(USE_CUDA)
	message(STATUS "Looking for CUDA toolkit")
	find_package(CUDA)
	add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

if(USE_ZLIB)
	include(zlib)
endif(USE_ZLIB)
if(USE_PNG)
	include(libpng)
endif(USE_PNG)

#if(USE_FREETYPE)
	include(opensans)
	include(freetype)
#endif(USE_FREETYPE)

include(libxml2)
include(v8)
add_definitions(-DLITEHTML_UTF8) # Force UTF on litehtml
include(litehtml)

# if FreeType is not requested we may use Xlib on Unix, or GDI on Windows... eventually.
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

include_directories(include common common/pcx demos/common core agg/src agg/include)
subdirs(agg/src common/pcx core renderers tools  demos)
