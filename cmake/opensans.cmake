# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.

include(functions)

download("http://dl.static.1001fonts.net/niceid/o/p/open-sans.zip" "open-sans.zip")
decompress_into("open-sans.zip" "open-sans")
add_custom_command(
	OUTPUT ${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Regular_ttf.h
	COMMAND xxd ARGS -i ${CMAKE_SOURCE_DIR}/open-sans/OpenSans-Regular.ttf > ${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Regular_ttf.h
	DEPENDS xxd
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/open-sans
	COMMENT "Generating Open Sans Arrays" VERBATIM)
