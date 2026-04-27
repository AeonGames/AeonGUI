# Copyright 2015,2026 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.
#
# Downloads at least one representative font for every CSS generic
# font family so the test suite (and demos) always have something
# for the FontDatabase to match against, regardless of the host's
# system fonts. All fonts below are OFL or Apache 2.0 licensed.
#
#   sans-serif  -> Open Sans   (Apache 2.0)
#   serif       -> Noto Serif  (OFL)
#   monospace   -> Roboto Mono (OFL)
#   cursive     -> Caveat      (OFL)
#   fantasy     -> Bungee      (OFL)
#
# Files land in <build>/bin/fonts/ which FontDatabase auto-scans on
# Initialize().

set(AEONGUI_FONT_DIR "${CMAKE_BINARY_DIR}/bin/fonts")
file(MAKE_DIRECTORY "${AEONGUI_FONT_DIR}")

# _aeongui_download_font(<label> <url> <output-path>)
#
# Downloads <url> into <output-path> if the file does not already
# exist. Removes a partial file and emits a warning on failure so
# subsequent configures retry without breaking the build.
function(_aeongui_download_font label url output)
    if(EXISTS "${output}")
        return()
    endif()
    message(STATUS "Downloading ${label}...")
    file(DOWNLOAD
        "${url}"
        "${output}"
        STATUS download_status
        SHOW_PROGRESS)
    list(GET download_status 0 download_status_code)
    if(download_status_code)
        file(REMOVE "${output}")
        message(WARNING "Failed to download ${label}: ${download_status}")
    endif()
endfunction()

# sans-serif
_aeongui_download_font(
    "Open Sans (Regular)"
    "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf"
    "${AEONGUI_FONT_DIR}/OpenSans[wdth,wght].ttf")

_aeongui_download_font(
    "Open Sans (Italic)"
    "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Italic%5Bwdth%2Cwght%5D.ttf"
    "${AEONGUI_FONT_DIR}/OpenSans-Italic[wdth,wght].ttf")

# serif
_aeongui_download_font(
    "Noto Serif"
    "https://github.com/google/fonts/raw/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf"
    "${AEONGUI_FONT_DIR}/NotoSerif[wdth,wght].ttf")

# monospace
_aeongui_download_font(
    "Roboto Mono"
    "https://github.com/google/fonts/raw/main/ofl/robotomono/RobotoMono%5Bwght%5D.ttf"
    "${AEONGUI_FONT_DIR}/RobotoMono[wght].ttf")

# cursive
_aeongui_download_font(
    "Caveat"
    "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"
    "${AEONGUI_FONT_DIR}/Caveat[wght].ttf")

# fantasy
_aeongui_download_font(
    "Bungee"
    "https://github.com/google/fonts/raw/main/ofl/bungee/Bungee-Regular.ttf"
    "${AEONGUI_FONT_DIR}/Bungee-Regular.ttf")
