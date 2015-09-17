# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the MIT License.
include(FindPNG)
if(NOT PNG_FOUND)
	if(NOT EXISTS "${CMAKE_SOURCE_DIR}/libpng-1.6.2.tar.gz")
		message(STATUS "libpng support requested but not found, please wait while the software package is downloaded...")
		file(DOWNLOAD "http://download.sourceforge.net/libpng/libpng-1.6.2.tar.gz" "${CMAKE_SOURCE_DIR}/libpng-1.6.2.tar.gz" STATUS libpng_dl_status LOG libpng_dl_log SHOW_PROGRESS)
		if(NOT libpng_dl_status MATCHES "0;\"no error\"")
			message("Download failed, did you set a proxy? ${libpng_dl_status}")
		endif(NOT libpng_dl_status MATCHES "0;\"no error\"")
		message(STATUS "Done downloading libpng")
	endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/libpng-1.6.2.tar.gz")

	if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
		message(STATUS "Extracting libpng-1.6.2.tar.bz2...")
		execute_process(COMMAND cmake -E tar xzvf libpng-1.6.2.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	endif(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
endif(NOT PNG_FOUND)
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
	set(SKIP_INSTALL_ALL ON CACHE INTERNAL "Using local png." FORCE)
	set(PNG_TESTS OFF CACHE INTERNAL "Using local png." FORCE)
	set(PNG_SHARED ON CACHE INTERNAL "Using local png." FORCE)
	set(PNG_STATIC ON CACHE INTERNAL "Using local png." FORCE)
	include_directories(libpng-1.6.2 ${CMAKE_BINARY_DIR}/libpng-1.6.2)
	add_subdirectory(libpng-1.6.2)
	set(PNG_LIBRARIES png16 CACHE INTERNAL "Using local png." FORCE)
	set(PNG_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/libpng-1.6.2" CACHE INTERNAL "Using local png." FORCE)
	set(PNG_LIBRARY ${PNG_LIBRARIES} CACHE INTERNAL "Using local png." FORCE)
	set(PNG_PNG_INCLUDE_DIR ${PNG_INCLUDE_DIRS} CACHE INTERNAL "Using local png." FORCE)
	set(PNG_FOUND ON CACHE INTERNAL "Using local png." FORCE)
endif(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libpng-1.6.2")
add_definitions(-DUSE_PNG ${PNG_DEFINITIONS})