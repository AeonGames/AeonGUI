# Copyright (C) 2016-2019 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

#
# Generate VS Code Environment
#
function(configure_vscode)
if(CMAKE_GENERATOR MATCHES "(MSYS|Unix) Makefiles")
  set(CODE_ZOOMLEVEL "0" CACHE STRING "window.zoomLevel for VS Code.")
  if(APPLE)
    set(DEBUGGER_TYPE "lldb")
    set(DEBUGGER_LAUNCH_EXTRA "")
  else()
    find_program(GDB_EXECUTABLE gdb HINTS ENV MINGW_PREFIX MSYS2_PATH)
    set(DEBUGGER_TYPE "cppdbg")
    set(DEBUGGER_EXECUTABLE "${GDB_EXECUTABLE}")
    set(DEBUGGER_MODE "gdb")
    set(DEBUGGER_SETUP_COMMANDS "[
                            {
                                \"description\": \"Enable pretty-printing for gdb\",
                                \"text\": \"-enable-pretty-printing\",
                                \"ignoreFailures\": true
                            },
                            {
                                \"description\": \"Enable all-exceptions\",
                                \"text\": \"catch throw\",
                                \"ignoreFailures\": true
                            }
                        ]")
                set(DEBUGGER_LAUNCH_EXTRA ",
                          \"miDebuggerPath\": \"${DEBUGGER_EXECUTABLE}\",
                          \"MIMode\": \"${DEBUGGER_MODE}\",
                          \"setupCommands\": ${DEBUGGER_SETUP_COMMANDS}")
  endif()
  set(DEBUG_PATH "${CMAKE_BINARY_DIR}/bin")
  if(CMAKE_GENERATOR MATCHES "MSYS Makefiles")
    set(DEBUG_PATH "${DEBUG_PATH};$ENV{MINGW_PREFIX}/bin")
    set(USE_EXTERNAL_CONSOLE "true")
  else()
    set(DEBUG_PATH "$ENV{PATH}:${DEBUG_PATH}")
    set(USE_EXTERNAL_CONSOLE "false")
  endif()

  set(DEBUG_CONFIGURATIONS "")

    #
    # Generate VS Code Environment
    #
    if("$ENV{MSYSTEM}" STREQUAL "MINGW64")
    set(VSCODE_DEFAULT_PROFILE_WINDOWS "MinGW GCC Bash")
    elseif("$ENV{MSYSTEM}" STREQUAL "UCRT64")
    set(VSCODE_DEFAULT_PROFILE_WINDOWS "MinGW UCRT64 Bash")
    elseif("$ENV{MSYSTEM}" STREQUAL "CLANG64")
    set(VSCODE_DEFAULT_PROFILE_WINDOWS "MinGW Clang Bash")
    else()
    set(VSCODE_DEFAULT_PROFILE_WINDOWS "PowerShell")
    endif()

  if(${ARGC})
  foreach(DIRECTORY ${ARGV})
      if(IS_DIRECTORY ${DIRECTORY} AND EXISTS ${DIRECTORY}/CMakeLists.txt)
          get_property(TARGETS DIRECTORY "${DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
          get_property(TESTS DIRECTORY "${DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
          foreach(TARGET ${TARGETS})
              get_target_property(target_type ${TARGET} TYPE)
              if (${target_type} STREQUAL "EXECUTABLE")
                  get_target_property(svg_files ${TARGET} AEONGUI_SVG_FILES)
                  if(svg_files)
                      foreach(SVG_FILE ${svg_files})
                          get_filename_component(SVG_NAME "${SVG_FILE}" NAME_WE)
                          message(STATUS "Generating debug launch configuration for ${TARGET} [${SVG_NAME}]")
                          set(DEBUG_CONFIGURATIONS "${DEBUG_CONFIGURATIONS}
                  {
                  \"name\": \"${TARGET} [${SVG_NAME}]\",
                  \"type\": \"${DEBUGGER_TYPE}\",
                  \"request\": \"launch\",
                  \"args\": [\"file:///${SVG_FILE}\"],
                  \"stopAtEntry\": false,
                  \"cwd\": \"${CMAKE_BINARY_DIR}\",
                  \"environment\": [
                      {
                          \"name\":\"PATH\",
                          \"value\":\"${DEBUG_PATH}\"
                      }                    
                  ],
                  \"externalConsole\": ${USE_EXTERNAL_CONSOLE},
                        \"program\": \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}\"${DEBUGGER_LAUNCH_EXTRA}
                    },\n")
                      endforeach()
                  else()
                      message(STATUS "Generating debug launch configuration for ${TARGET}")
                      set(DEBUG_CONFIGURATIONS "${DEBUG_CONFIGURATIONS}
                  {
                  \"name\": \"${TARGET}\",
                  \"type\": \"${DEBUGGER_TYPE}\",
                  \"request\": \"launch\",
                  \"args\": [],
                  \"stopAtEntry\": false,
                  \"cwd\": \"${CMAKE_BINARY_DIR}\",
                  \"environment\": [
                      {
                          \"name\":\"PATH\",
                          \"value\":\"${DEBUG_PATH}\"
                      }                    
                  ],
                  \"externalConsole\": ${USE_EXTERNAL_CONSOLE},
                        \"program\": \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}\"${DEBUGGER_LAUNCH_EXTRA}
                    },\n")
                  endif()
              endif()
          endforeach(TARGET)
      endif()
  endforeach()
  endif()

  configure_file("${PROJECT_SOURCE_DIR}/cmake/cmake-kits.json.in"
                 "${PROJECT_SOURCE_DIR}/.vscode/cmake-kits.json")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/settings.json.in"
                 "${PROJECT_SOURCE_DIR}/.vscode/settings.json")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/launch.json.in"
                 "${PROJECT_SOURCE_DIR}/.vscode/launch.json")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/tasks.json.in"
                 "${PROJECT_SOURCE_DIR}/.vscode/tasks.json")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/c_cpp_properties.json.in"
                 "${PROJECT_SOURCE_DIR}/.vscode/c_cpp_properties.json" @ONLY)
endif()
endfunction()
