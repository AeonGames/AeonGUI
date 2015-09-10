# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/libcss-0.5.0-src.tar.gz")
	message(STATUS "Downloading libcss...")
	file(DOWNLOAD "http://download.netsurf-browser.org/libs/releases/libcss-0.5.0-src.tar.gz" "${CMAKE_SOURCE_DIR}/libcss-0.5.0-src.tar.gz" STATUS libcss_dl_status LOG libcss_dl_log SHOW_PROGRESS)
	if(NOT libcss_dl_status MATCHES "0;\"No error\"")
		message("Download failed, did you set a proxy? ${libcss_dl_status}")
		file(REMOVE "${CMAKE_SOURCE_DIR}/libcss-0.5.0-src.tar.gz")
	endif()
	message(STATUS "Done downloading libcss")
endif()
if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libcss-0.5.0")
	message(STATUS "Extracting libcss...")
	execute_process(COMMAND cmake -E tar xzvf libcss-0.5.0-src.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} ERROR_VARIABLE extract_result)
	message(STATUS "Extract Result ${extract_result}")
else()
	#TODO Set projects
endif()
