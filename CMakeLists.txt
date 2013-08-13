cmake_minimum_required(VERSION 2.8)
if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT(AeonGUI)
INCLUDE(FindOpenGL)

OPTION(USE_FREETYPE  "Use FreeType2 for font rendering")
OPTION(USE_ZLIB "Use the zlib compression library")
OPTION(USE_PNG "Enable PNG image support (Requires zlib)")
OPTION(USE_CUDA "Enable CUDA support")

OPTION(RELEASE_BUILD "Build for Release")

set(HTTP_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")

if(USE_CUDA)
message(STATUS "Looking for CUDA toolkit")
find_package(CUDA)
add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

if(USE_FREETYPE)
add_definitions(-DUSE_FREETYPE)
include(FindFreetype)
if(FREETYPE_FOUND)
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2})
else(FREETYPE_FOUND)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/freetype-2.4.12.tar.bz2")
message(STATUS "FreeType2 support requested but not found, please wait while the software package is downloaded...")
set(ENV{http_proxy} "${HTTP_PROXY}")
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

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/freetype-2.4.12/include)
set(FREETYPE_LIBRARIES freetype)
set(FT_SOURCES
      # This list of source files has been extracted from the file INSTALL.ANY
      # at the freetype source distribution and is used to create a custom build
      # of the library since by itself does not provide a consistent way to build
      # 64 bit versions of the library on Windows using Visual Studio/Visual C.
      # Most Linux distributions will already have development libraries for freetype,
      # but if not or a custom build is desired, this target project should suffice.
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftsystem.c
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftinit.c
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftdebug.c

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftbase.c

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftbbox.c       #-- recommended, see <freetype/ftbbox.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftglyph.c      #-- recommended, see <freetype/ftglyph.h>

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftbdf.c        #-- optional, see <freetype/ftbdf.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftbitmap.c     #-- optional, see <freetype/ftbitmap.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftcid.c        #-- optional, see <freetype/ftcid.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftfstype.c     #-- optional
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftgasp.c       #-- optional, see <freetype/ftgasp.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftgxval.c      #-- optional, see <freetype/ftgxval.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftlcdfil.c     #-- optional, see <freetype/ftlcdfil.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftmm.c         #-- optional, see <freetype/ftmm.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftotval.c      #-- optional, see <freetype/ftotval.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftpatent.c     #-- optional
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftpfr.c        #-- optional, see <freetype/ftpfr.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftstroke.c     #-- optional, see <freetype/ftstroke.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftsynth.c      #-- optional, see <freetype/ftsynth.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/fttype1.c      #-- optional, see <freetype/t1tables.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftwinfnt.c     #-- optional, see <freetype/ftwinfnt.h>
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftxf86.c       #-- optional, see <freetype/ftxf86.h>

      #${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/base/ftmac.c        #-- only on the Macintosh

    #-- font drivers (optional; at least one is needed)

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/bdf/bdf.c           #-- BDF font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/cff/cff.c           #-- CFF/OpenType font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/cid/type1cid.c      #-- Type 1 CID-keyed font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/pcf/pcf.c           #-- PCF font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/pfr/pfr.c           #-- PFR/TrueDoc font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/sfnt/sfnt.c         #-- SFNT files support
                                                                  #   (TrueType & OpenType)
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/truetype/truetype.c #-- TrueType font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/type1/type1.c       #-- Type 1 font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/type42/type42.c     #-- Type 42 font driver
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/winfonts/winfnt.c   #-- Windows FONT / FNT font driver

    #-- rasterizers (optional; at least one is needed for vector formats)

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/raster/raster.c     #-- monochrome rasterizer
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/smooth/smooth.c     #-- anti-aliasing rasterizer

    #-- auxiliary modules (optional)

      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/autofit/autofit.c   #-- auto hinting module
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/cache/ftcache.c     #-- cache sub-system (in beta)
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/gzip/ftgzip.c       #-- support for compressed fonts (.gz)
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/lzw/ftlzw.c         #-- support for compressed fonts (.Z)
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/bzip2/ftbzip2.c     #-- support for compressed fonts (.bz2)
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/gxvalid/gxvalid.c   #-- TrueTypeGX/AAT table validation
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/otvalid/otvalid.c   #-- OpenType table validation
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/psaux/psaux.c       #-- PostScript Type 1 parsing
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/pshinter/pshinter.c #-- PS hinting module
      ${CMAKE_SOURCE_DIR}/freetype-2.4.12/src/psnames/psnames.c   #-- PostScript glyph names support
    )

add_library(freetype ${FT_SOURCES})
set_target_properties(freetype PROPERTIES COMPILE_DEFINITIONS "FT2_BUILD_LIBRARY")
#set_target_properties(freetype PROPERTIES COMPILE_FLAGS "-DFT2_BUILD_LIBRARY")

endif(FREETYPE_FOUND)
endif(USE_FREETYPE)

IF(USE_ZLIB OR USE_PNG)
INCLUDE(FindZLIB)
IF(NOT ZLIB_FOUND)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zlib-1.2.8.tar.gz")
message(STATUS "zlib support requested but not found, please wait while the software package is downloaded...")
set(ENV{http_proxy} "${HTTP_PROXY}")
file(DOWNLOAD "http://zlib.net/zlib-1.2.8.tar.gz" "${CMAKE_CURRENT_SOURCE_DIR}/zlib-1.2.8.tar.gz" STATUS zlib_dl_status LOG zlib_dl_log SHOW_PROGRESS)
if(NOT zlib_dl_status MATCHES "0;\"no error\"")
message("Download failed, did you set a proxy? ${zlib_dl_status}")
endif(NOT zlib_dl_status MATCHES "0;\"no error\"")
message(STATUS "Done downloading zlib")
endif(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zlib-1.2.8.tar.gz")

IF(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/zlib-1.2.8")
MESSAGE(STATUS "Extracting zlib-1.2.8.tar.gz...")
EXECUTE_PROCESS(COMMAND cmake -E tar xzvf zlib-1.2.8.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
ENDIF(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/zlib-1.2.8")
INCLUDE_DIRECTORIES(zlib-1.2.8 ${CMAKE_BINARY_DIR}/zlib-1.2.8)
ADD_SUBDIRECTORY(zlib-1.2.8)
SET(ZLIB_INCLUDE_DIR zlib-1.2.8)
SET(ZLIB_LIBRARIES zlib)
ENDIF(NOT ZLIB_FOUND)
ENDIF(USE_ZLIB OR USE_PNG)
IF(USE_ZLIB)
ADD_DEFINITIONS(-DUSE_ZLIB)
ENDIF(USE_ZLIB)
IF(USE_PNG)
IF(NOT ZLIB_FOUND)
SET(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
ENDIF(NOT ZLIB_FOUND)
INCLUDE(FindPNG)
IF(NOT PNG_FOUND)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libpng-1.6.2.tar.gz")
message(STATUS "libpng support requested but not found, please wait while the software package is downloaded...")
set(ENV{http_proxy} "${HTTP_PROXY}")
file(DOWNLOAD "http://download.sourceforge.net/libpng/libpng-1.6.2.tar.gz" "${CMAKE_CURRENT_SOURCE_DIR}/libpng-1.6.2.tar.gz" STATUS libpng_dl_status LOG libpng_dl_log SHOW_PROGRESS)
if(NOT libpng_dl_status MATCHES "0;\"no error\"")
message("Download failed, did you set a proxy? ${libpng_dl_status}")
endif(NOT libpng_dl_status MATCHES "0;\"no error\"")
message(STATUS "Done downloading libpng")
endif(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libpng-1.6.2.tar.gz")

IF(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
MESSAGE(STATUS "Extracting libpng-1.6.2.tar.gz...")
EXECUTE_PROCESS(COMMAND cmake -E tar xjvf libpng-1.6.2.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
ENDIF(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
SET(SKIP_INSTALL_ALL true)
SET(PNG_TESTS false)
SET(PNG_SHARED true)
SET(PNG_STATIC true)
INCLUDE_DIRECTORIES(libpng-1.6.2 ${CMAKE_BINARY_DIR}/libpng-1.6.2)
ADD_SUBDIRECTORY(libpng-1.6.2)
SET(PNG_LIBRARIES png16)
ENDIF(NOT PNG_FOUND)
ADD_DEFINITIONS(-DUSE_PNG ${PNG_DEFINITIONS})
ENDIF(USE_PNG)

# I will be adding Java bindings in the near future, just check for the JNI and java (compiler?) for now.
#INCLUDE(FindJNI)
#IF(JNI_FOUND)
#INCLUDE(FindJava)
#ENDIF(JNI_FOUND)
# if FreeType is not requested we may use Xlib on Unix, or GDI on Windows... eventually.
INCLUDE(FindX11)
IF(CMAKE_COMPILER_IS_GNUCXX)
# -std=c++0x is required to be able to include stdint.h as cstdint, weird, I know.
# But then CUDA doesn't like c++0x, so stick to .h files until it does *sigh*
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
IF(RELEASE_BUILD)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -fomit-frame-pointer -O3")
ELSE(RELEASE_BUILD)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
ENDIF(RELEASE_BUILD)
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
INCLUDE_DIRECTORIES(include common common/pcx demos/common core)
SUBDIRS(common/pcx core renderers tools demos)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
