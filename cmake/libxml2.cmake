# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.

#LIBXML2 Version 2.9.2 has a broken MSVC build system (missing configure.in from the distro)
set(XML2_VERSION 2.9.1)
set(LIBXML_CONFIG_PARAMS "")
include(FindLibXml2)
if(NOT LIBXML2_FOUND)
	if(NOT EXISTS "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}.tar.gz")
		message(STATUS "libxml2 support requested but not found, please wait while the software package is downloaded...")
		set(ENV{http_proxy} "${HTTP_PROXY}")
		file(DOWNLOAD "ftp://xmlsoft.org/libxml2/libxml2-${XML2_VERSION}.tar.gz" "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}.tar.gz" STATUS libxml2_dl_status LOG libxml2_dl_log SHOW_PROGRESS)
		if(NOT libxml2_dl_status MATCHES "0;")
			message("Download failed, did you set a proxy? ${libxml2_dl_status}")
		endif(NOT libxml2_dl_status MATCHES "0;")
		message(STATUS "Done downloading libxml2")
	endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}.tar.gz")
	if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}")
		message(STATUS "Extracting libxml2...")
		execute_process(COMMAND cmake -E tar xzvf libxml2-${XML2_VERSION}.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
		# Patch Windows config file
		file(READ ${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}/include/win32config.h XML2_WIN32CONFIG)
		string(REPLACE "#define snprintf _snprintf" "#if _MSC_VER < 1900\n#define snprintf _snprintf\n#endif" 
				XML2_WIN32CONFIG_PATCHED "${XML2_WIN32CONFIG}")
		file(WRITE ${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}/include/win32config.h "${XML2_WIN32CONFIG_PATCHED}") 
	endif(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}")
endif(NOT LIBXML2_FOUND)

if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}")	
	if(MSVC)
		set(LIBXML2_LIBRARIES "${CMAKE_BINARY_DIR}/libxml2/lib/libxml2.lib" CACHE FILEPATH "LibXml2 Library" FORCE)
		set(LIBXML2_INCLUDE_DIR "${CMAKE_BINARY_DIR}/libxml2/include/libxml2" CACHE PATH "LibXml2 include directory" FORCE)
		set(LIBXML2_XMLLINT_EXECUTABLE "${CMAKE_BINARY_DIR}/libxml2/bin/xmllint.exe" CACHE FILEPATH "LibXml2 include directory" FORCE)
		string(REGEX REPLACE "/" "\\\\" WIN_CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
		message(STATUS "Configuring libxml2...")
		if(USE_ZLIB)
			set(LIBXML_CONFIG_PARAMS ${LIBXML_CONFIG_PARAMS} zlib=yes)
			message(STATUS "LIBXML_CONFIG_PARAMS ${LIBXML_CONFIG_PARAMS}")
		endif()
		add_custom_target(libxml2
			COMMAND cscript configure.js debug=yes iconv=no ${LIBXML_CONFIG_PARAMS} prefix=${WIN_CMAKE_BINARY_DIR}\\libxml2
			COMMAND nmake install 
			BYPRODUCTS ${LIBXML2_LIBRARIES}
			WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/libxml2-${XML2_VERSION}/win32" COMMENT "Building LibXml2" VERBATIM)
	endif(MSVC)
endif()