# Copyright (C) 2021 Rodrigo Jose Hernandez Cordoba
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

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
parser_code(webidl ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(WEBIDL_SOURCES webidl.cpp)
set(WEBIDL_HEADERS webidl.h WebIDLType.h Attribute.h)
add_executable(webidl
    ${WEBIDL_SOURCES}
    ${WEBIDL_HEADERS}
    ${BISON_webidl_parser_OUTPUTS}
    ${FLEX_webidl_lexer_OUTPUTS}
)
