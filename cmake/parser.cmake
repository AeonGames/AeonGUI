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

find_package(BISON)
find_package(FLEX)

function(parser_code parser_name source_path output_path)
    if(FLEX_FOUND AND BISON_FOUND)
        message(STATUS "Parser for ${parser_name} code will be generated from Flex and Bison code.")
        message(STATUS "Build the 'update-${parser_name}-parser-code' target if you want to update the pre-generated code.")

        bison_target(${parser_name}_parser ${source_path}/${parser_name}.ypp ${CMAKE_CURRENT_BINARY_DIR}/${parser_name}_parser.cpp COMPILE_FLAGS "-Wother -Wcounterexamples -v -d")
        flex_target(${parser_name}_lexer ${source_path}/${parser_name}.l ${CMAKE_CURRENT_BINARY_DIR}/${parser_name}_lexer.cpp COMPILE_FLAGS "-Cf")
        add_flex_bison_dependency(${parser_name}_lexer ${parser_name}_parser)
        if(MSVC)
            set_source_files_properties(${FLEX_${parser_name}_lexer_OUTPUTS} PROPERTIES COMPILE_FLAGS -FIcstdint)
        endif()

        add_custom_target(update-${parser_name}-parser-code
        COMMAND
            ${CMAKE_COMMAND} -E copy ${BISON_${parser_name}_parser_OUTPUTS} ${FLEX_${parser_name}_lexer_OUTPUTS} ${output_path}/
        DEPENDS
            ${parser_name}_parser
            ${parser_name}_lexer
        COMMENT "Updating pre-generated ${parser_name} parser code.")
    else()
        message(STATUS "Using pre-generated parser code for ${parser_name} parser.")
        message(STATUS "Install Flex and Bison if you want to regenerate the parser code at build time.")
        set(BISON_${parser_name}_parser_OUTPUTS
            ${output_path}/${parser_name}_parser.hpp
            ${output_path}/${parser_name}_parser.cpp
        )
        set(FLEX_${parser_name}_lexer_OUTPUTS
            ${output_path}/${parser_name}_lexer.cpp
        )
        if(MSVC)
            set_source_files_properties(${BISON_${parser_name}_parser_OUTPUTS} ${FLEX_${parser_name}_lexer_OUTPUTS} PROPERTIES COMPILE_FLAGS -FIcstdint)
        endif()
    endif()
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set_source_files_properties(${BISON_${parser_name}_parser_OUTPUTS} PROPERTIES COMPILE_FLAGS -Wno-free-nonheap-object)
    endif()

    set(BISON_${parser_name}_parser_OUTPUTS ${BISON_${parser_name}_parser_OUTPUTS} CACHE STRING "" FORCE)
    mark_as_advanced(BISON_${parser_name}_parser_OUTPUTS)

    set(FLEX_${parser_name}_lexer_OUTPUTS ${FLEX_${parser_name}_lexer_OUTPUTS} CACHE STRING "" FORCE)
    mark_as_advanced(FLEX_${parser_name}_lexer_OUTPUTS)
endfunction()
