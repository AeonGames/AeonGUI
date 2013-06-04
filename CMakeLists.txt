cmake_minimum_required(VERSION 2.8)
if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT(AeonGUI)
INCLUDE(FindOpenGL)

OPTION(USE_FREETYPE  "Use FreeType2 for font rendering")

OPTION(RELEASE_BUILD "Build for Release")

OPTION(USE_HASH_CRC  "Use CRC32 for Resource hashing")
OPTION(USE_HASH_SHA1 "Use SHA1 for Resource hashing")
OPTION(USE_HASH_MD5 "Use MD5 for Resource hashing")

set(HTTP_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")

IF((USE_HASH_CRC AND USE_HASH_MD5) OR (USE_HASH_CRC AND USE_HASH_SHA1) OR (USE_HASH_MD5 AND USE_HASH_SHA1))
MESSAGE(FATAL_ERROR "Must chose only one hashing algorithm")
ENDIF((USE_HASH_CRC AND USE_HASH_MD5) OR (USE_HASH_CRC AND USE_HASH_SHA1) OR (USE_HASH_MD5 AND USE_HASH_SHA1))

if(USE_FREETYPE)
ADD_DEFINITIONS(-DUSE_FREETYPE)
include(FindFreetype)
if(FREETYPE_FOUND)
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2})
else(FREETYPE_FOUND)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/freetype-2.4.12.tar.bz2")
message(STATUS "FreeType2 support requested but not found, please wait while the software package is downloaded...")
if(HTTP_PROXY MATCHES "^$")
set(ENV{http_proxy} "${HTTP_PROXY}")
endif(HTTP_PROXY MATCHES "^$")
file(DOWNLOAD "http://download.savannah.gnu.org/releases/freetype/freetype-2.4.12.tar.bz2" "${CMAKE_CURRENT_SOURCE_DIR}/freetype-2.4.12.tar.bz2" STATUS ft_dl_status LOG ft_dl_log SHOW_PROGRESS)
if(NOT ft_dl_status MATCHES "0;\"no error\"")
message("Download failed, did you set a proxy? ${ft_dl_status}")
endif(NOT ft_dl_status MATCHES "0;\"no error\"")
message(STATUS "Done downloading FreeType2")
endif(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/freetype-2.4.12.tar.bz2")

if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")
MESSAGE(STATUS "Extracting freetype-2.4.12.tar.bz2...")
EXECUTE_PROCESS(COMMAND cmake -E tar xjvf freetype-2.4.12.tar.bz2 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
EXECUTE_PROCESS(COMMAND ${CMAKE_MAKE_PROGRAM} builds/win32/vc2010/freetype.sln /upgrade WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")
endif(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/freetype-2.4.12/include ${CMAKE_SOURCE_DIR}/freetype-2.4.12/include/freetype)
IF(MSVC)
set(FREETYPE_LIBRARIES ${CMAKE_SOURCE_DIR}/freetype-2.4.12/objs/win32/vc2010/freetype2412.lib)
ADD_CUSTOM_TARGET(freetype2 ALL ${CMAKE_MAKE_PROGRAM} builds/win32/vc2010/freetype.sln /build Release WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12" COMMENT "Building FreeType 2" VERBATIM)
ELSE(MSVC)
ADD_CUSTOM_TARGET(freetype2 ALL ${CMAKE_MAKE_PROGRAM} dependencies COMMAND ${CMAKE_MAKE_PROGRAM} native WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12" COMMENT "Building FreeType 2" VERBATIM)
ENDIF(MSVC)

endif(FREETYPE_FOUND)
endif(USE_FREETYPE)

# I will be adding Java bindings in the near future, just check for the JNI and java (compiler?) for now.
#INCLUDE(FindJNI)
#IF(JNI_FOUND)
#INCLUDE(FindJava)
#ENDIF(JNI_FOUND)
# if FreeType is not found we'll use Xlib on Unix, or GDI on Windows...
# eventually
INCLUDE(FindX11)
IF(CMAKE_COMPILER_IS_GNUCXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
IF(RELEASE_BUILD)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -fomit-frame-pointer -O3")
ELSE(RELEASE_BUILD)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
ENDIF(RELEASE_BUILD)
# -std=c++0x is required to be able to include stdint.h as cstdint, weird, I know.
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF(CMAKE_COMPILER_IS_GNUC)
SET(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -fomit-frame-pointer -O3")
SET(CMAKE_C_FLAGS_DEBUG "-DDEBUG -g -O0")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
ENDIF(CMAKE_COMPILER_IS_GNUC)

IF(MSVC)
ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
ENDIF(MSVC)

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/demos/common ${CMAKE_CURRENT_SOURCE_DIR}/core)
INCLUDE_DIRECTORIES(include common demos/common core)
SUBDIRS(core renderers tools demos)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
