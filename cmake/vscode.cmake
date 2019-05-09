#
# Generate VS Code Environment
#
if(CMAKE_GENERATOR MATCHES "(MSYS|Unix) Makefiles")
  find_program(BASH_EXECUTABLE bash HINTS ENV MINGW_PREFIX MSYS2_PATH)
  set(CODE_ZOOMLEVEL "0" CACHE STRING "window.zoomLevel for VS Code.")
  find_program(GDB_EXECUTABLE gdb HINTS ENV MINGW_PREFIX MSYS2_PATH)
  set(DEBUG_PATH "${CMAKE_BINARY_DIR}/bin")
  if(CMAKE_GENERATOR MATCHES "MSYS Makefiles")
    set(DEBUG_PATH "${DEBUG_PATH};$ENV{MINGW_PREFIX}/bin")
    set(USE_EXTERNAL_CONSOLE "true")
  else()
    set(DEBUG_PATH "$ENV{PATH}:${DEBUG_PATH}")
    set(USE_EXTERNAL_CONSOLE "false")
  endif()
  configure_file("${CMAKE_SOURCE_DIR}/cmake/settings.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/settings.json")
  configure_file("${CMAKE_SOURCE_DIR}/cmake/launch.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/launch.json")
  configure_file("${CMAKE_SOURCE_DIR}/cmake/c_cpp_properties.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/c_cpp_properties.json" @ONLY)
endif()
