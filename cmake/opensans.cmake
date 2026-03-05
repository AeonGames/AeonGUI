# Copyright 2015,2026 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.
#
# Open Sans is licensed under the Apache License, Version 2.0.
# https://github.com/google/fonts/blob/main/ofl/opensans/OFL.txt

set(OPENSANS_FONT_DIR "${CMAKE_BINARY_DIR}/bin/fonts")
file(MAKE_DIRECTORY "${OPENSANS_FONT_DIR}")

set(OPENSANS_REGULAR "${OPENSANS_FONT_DIR}/OpenSans[wdth,wght].ttf")
set(OPENSANS_ITALIC  "${OPENSANS_FONT_DIR}/OpenSans-Italic[wdth,wght].ttf")

if(NOT EXISTS "${OPENSANS_REGULAR}")
    message(STATUS "Downloading Open Sans (Regular)...")
    file(DOWNLOAD
        "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf"
        "${OPENSANS_REGULAR}"
        STATUS download_status
        SHOW_PROGRESS)
    list(GET download_status 0 download_status_code)
    if(download_status_code)
        file(REMOVE "${OPENSANS_REGULAR}")
        message(WARNING "Failed to download Open Sans Regular: ${download_status}")
    endif()
endif()

if(NOT EXISTS "${OPENSANS_ITALIC}")
    message(STATUS "Downloading Open Sans (Italic)...")
    file(DOWNLOAD
        "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Italic%5Bwdth%2Cwght%5D.ttf"
        "${OPENSANS_ITALIC}"
        STATUS download_status
        SHOW_PROGRESS)
    list(GET download_status 0 download_status_code)
    if(download_status_code)
        file(REMOVE "${OPENSANS_ITALIC}")
        message(WARNING "Failed to download Open Sans Italic: ${download_status}")
    endif()
endif()
