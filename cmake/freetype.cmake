# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the MIT License.
include(FindFreetype)
if(FREETYPE_FOUND)
	INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_ft2build} ${FREETYPE_INCLUDE_DIR_freetype2})
else(FREETYPE_FOUND)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/freetype-2.4.12.tar.bz2")
message(STATUS "Please wait while the freetype source package is downloaded...")
set(ENV{http_proxy} "${HTTP_PROXY}")
file(DOWNLOAD "http://download.savannah.gnu.org/releases/freetype/freetype-2.4.12.tar.bz2" "${CMAKE_SOURCE_DIR}/freetype-2.4.12.tar.bz2" STATUS ft_dl_status LOG ft_dl_log SHOW_PROGRESS)
if(NOT ft_dl_status MATCHES "0;\"no error\"")
message("Download failed, did you set a proxy? ${ft_dl_status}")
endif(NOT ft_dl_status MATCHES "0;\"no error\"")
message(STATUS "Done downloading FreeType2")
endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/freetype-2.4.12.tar.bz2")

if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")
MESSAGE(STATUS "Extracting freetype-2.4.12.tar.bz2...")
EXECUTE_PROCESS(COMMAND cmake -E tar xjvf freetype-2.4.12.tar.bz2 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
EXECUTE_PROCESS(COMMAND ${CMAKE_MAKE_PROGRAM} builds/win32/vc2010/freetype.sln /upgrade WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")
endif(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/freetype-2.4.12")

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

add_library(freetype2 ${FT_SOURCES})
set_target_properties(freetype2 PROPERTIES COMPILE_DEFINITIONS "FT2_BUILD_LIBRARY")
include_directories(${CMAKE_SOURCE_DIR}/freetype-2.4.12/include)

set(FREETYPE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/freetype-2.4.12/include" CACHE INTERNAL "FreeType2 include directories" FORCE)
set(FREETYPE_FT2BUILD_INCLUDE_DIR "${FREETYPE_INCLUDE_DIRS}" CACHE INTERNAL "FreeType2 build include directory" FORCE)
set(FREETYPE_LIBRARY freetype2 CACHE INTERNAL "FreeType2 library" FORCE)
set(FREETYPE_FOUND ON CACHE INTERNAL "Using localy compiled FT2" FORCE)

endif(FREETYPE_FOUND)
