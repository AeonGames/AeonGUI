# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2 License.

find_package(Freetype)
if((NOT FREETYPE_FOUND) OR (MSVC))
	include(functions)
	download("http://download.savannah.gnu.org/releases/freetype/freetype-2.6.tar.gz" "freetype-2.6.tar.gz")
	decompress("freetype-2.6.tar.gz" "freetype-2.6")
	add_subdirectory("${CMAKE_SOURCE_DIR}/freetype-2.6" "${CMAKE_BINARY_DIR}/freetype-2.6")	
	set(FREETYPE_INCLUDE_DIR_freetype2 "${CMAKE_SOURCE_DIR}/freetype-2.6/include" CACHE PATH "FreeType2 Include Directory")
	set(FREETYPE_INCLUDE_DIR_ft2build  "${CMAKE_SOURCE_DIR}/freetype-2.6/include" CACHE PATH "ft2build Include Directory")
	set(FREETYPE_LIBRARIES "freetype" CACHE INTERNAL "freetype library")
endif()
