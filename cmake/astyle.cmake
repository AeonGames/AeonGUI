# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the MIT License.
if(WIN32)
# Download Windows Astyle binary.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/AStyle_2.04_windows.zip")
message(STATUS "Please wait while downloading the Astyle code beautifier...")
set(ENV{http_proxy} "${HTTP_PROXY}")
file(DOWNLOAD "http://iweb.dl.sourceforge.net/project/astyle/astyle/astyle%202.04/AStyle_2.04_windows.zip" "${CMAKE_SOURCE_DIR}/AStyle_2.04_windows.zip" STATUS astyle_dl_status LOG astyle_dl_log SHOW_PROGRESS)
if(NOT astyle_dl_status MATCHES "0;\"No error\"")
file(REMOVE "${CMAKE_SOURCE_DIR}/AStyle_2.04_windows.zip")
message(FATAL_ERROR "Download failed, did you set a proxy? STATUS: ${astyle_dl_status}")
endif(NOT astyle_dl_status MATCHES "0;\"No error\"")
message(STATUS "Done downloading Astyle binary")
endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/AStyle_2.04_windows.zip")

# Extract Astyle.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/AStyle.exe")
message(STATUS "Extracting Astyle...")
execute_process(COMMAND cmake -E tar xzvf AStyle_2.04_windows.zip WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} ERROR_VARIABLE extract_result)
execute_process(COMMAND cmake -E copy AStyle/bin/AStyle.exe AStyle.exe WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} ERROR_VARIABLE extract_result)
endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/AStyle.exe")
endif(WIN32)
