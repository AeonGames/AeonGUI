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

define_property(TARGET PROPERTY DEBUG_ARGUMENTS INHERITED
    BRIEF_DOCS "Debug Arguments"
    FULL_DOCS "Arguments to be passed when debugging")

#
# Generate VS Code Environment
#
function(configure_vscode)
if(CMAKE_GENERATOR MATCHES "(MSYS|Unix) Makefiles")
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

  set(DEBUG_CONFIGURATIONS "")

  if(${ARGC})
  foreach(DIRECTORY ${ARGV})
      if(IS_DIRECTORY ${DIRECTORY} AND EXISTS ${DIRECTORY}/CMakeLists.txt)
          get_property(TARGETS DIRECTORY "${DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
          foreach(TARGET ${TARGETS})
              get_target_property(target_type ${TARGET} TYPE)
              if (${target_type} STREQUAL "EXECUTABLE")
                  get_property(debug_arguments TARGET ${TARGET} PROPERTY DEBUG_ARGUMENTS)
                  message(STATUS "Generating debug launch configuration for ${TARGET}")
                  set(DEBUG_CONFIGURATIONS "${DEBUG_CONFIGURATIONS}
                  {
                  \"name\": \"${TARGET}\",
                  \"type\": \"cppdbg\",
                  \"request\": \"launch\",
                  \"args\": [\"${debug_arguments}\"],
                  \"stopAtEntry\": false,
                  \"cwd\": \"${CMAKE_BINARY_DIR}\",
                  \"environment\": [
                      {
                          \"name\":\"PATH\",
                          \"value\":\"${DEBUG_PATH}\"
                      }                    
                  ],
                  \"externalConsole\": true,
                        \"program\": \"${CMAKE_BINARY_DIR}/bin/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}\",
                        \"miDebuggerPath\": \"${GDB_EXECUTABLE}\",
                        \"MIMode\": \"gdb\",
                        \"setupCommands\": [
                        {
                        \"description\": \"Enable pretty-printing for gdb\",
                        \"text\": \"enable pretty printing\",
                        \"ignoreFailures\": false
                        },
                        {
                        \"description\": \"Break on Throw\",
                        \"text\": \"catch throw\",
                        \"ignoreFailures\": false
                        }
                  ]
                  },\n")
              endif()
          endforeach(TARGET)
      endif()
  endforeach()
  endif()

  configure_file("${CMAKE_SOURCE_DIR}/cmake/settings.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/settings.json")
  configure_file("${CMAKE_SOURCE_DIR}/cmake/launch.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/launch.json")
  configure_file("${CMAKE_SOURCE_DIR}/cmake/c_cpp_properties.json.in"
                 "${CMAKE_SOURCE_DIR}/.vscode/c_cpp_properties.json" @ONLY)
endif()
endfunction()
