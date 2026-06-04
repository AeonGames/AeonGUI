# Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba
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

# xmlcxx_generate(<out_var> <input> [CLASS <name>] [NAMESPACE <ns>])
#
# Runs the xmlcxx tool at build time to compile an XHTML/SVG document into C++
# source (a .cpp and matching .hpp) under the current binary directory, and
# returns the generated source files in <out_var>. Add the result to a target
# and link it against AeonGUI.
#
#   include(xmlcxx)
#   xmlcxx_generate(FPS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/fps.xhtml CLASS FpsDocument)
#   add_executable(demo main.cpp ${FPS_SRCS})
#   target_link_libraries(demo PRIVATE AeonGUI)
function(xmlcxx_generate out_var input)
    cmake_parse_arguments(XMLCXX "" "CLASS;NAMESPACE" "" ${ARGN})

    get_filename_component(_stem "${input}" NAME_WE)
    set(_cpp "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.xmlcxx.cpp")
    set(_hpp "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.xmlcxx.hpp")

    set(_args "${input}" -o "${_cpp}" --header "${_hpp}")
    if(XMLCXX_CLASS)
        list(APPEND _args --class "${XMLCXX_CLASS}")
    endif()
    if(XMLCXX_NAMESPACE)
        list(APPEND _args --namespace "${XMLCXX_NAMESPACE}")
    endif()

    add_custom_command(
        OUTPUT "${_cpp}" "${_hpp}"
        COMMAND xmlcxx ${_args}
        DEPENDS xmlcxx "${input}"
        COMMENT "xmlcxx: generating ${_stem} from ${input}"
        VERBATIM
    )

    set(${out_var} "${_cpp}" "${_hpp}" PARENT_SCOPE)
endfunction()
