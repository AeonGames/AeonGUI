# Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba
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

include(functions)

function(aeongames_check_py_module import_name install_name)
    file(WRITE ${CMAKE_BINARY_DIR}/find_${import_name}.py "import imp\ntry:\n\timp.find_module('${import_name}')\n\tprint 'MODULE-FOUND'\nexcept ImportError:\n\tprint 'MODULE-NOTFOUND'\n")
    execute_process ( COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/find_${import_name}.py OUTPUT_VARIABLE MODULE OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT MODULE)
        message(STATUS "${import_name} module not found, attempting install")
        execute_process ( COMMAND ${PYTHON_EXECUTABLE} -m pip install ${install_name} RESULT_VARIABLE PIP_INSTALL_RESULT)
        if(NOT PIP_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "${import_name} instalation failed. Try manual instalation. Interpreter found was: ${PYTHON_EXECUTABLE}")
        endif()
    endif()
endfunction()

function(aeongui_configure_duktape)

    download("https://duktape.org/duktape-2.3.0.tar.xz" "duktape-2.3.0.tar.xz")
    decompress("duktape-2.3.0.tar.xz" "duktape-2.3.0")

    find_package (PythonInterp 2.7 REQUIRED)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "You need a Python 2 Interpreter installed in order to generate duktape code.")
    endif()
    if(NOT IS_DIRECTORY "${CMAKE_BINARY_DIR}/duktape")
        aeongames_check_py_module("yaml" "pyyaml")
        execute_process(
            COMMAND 
                ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/duktape-2.3.0/tools/configure.py
                    --source-directory ${CMAKE_SOURCE_DIR}/duktape-2.3.0/src-input
                    --output-directory ${CMAKE_BINARY_DIR}/duktape
                    --config-metadata ${CMAKE_SOURCE_DIR}/duktape-2.3.0/config
                    -DDUK_USE_FASTINT 
                    -DDUK_USE_FATAL_HANDLER
                    --dll
            RESULT_VARIABLE DUKTAPE_CONFIG_RESULT)
            if(NOT DUKTAPE_CONFIG_RESULT EQUAL 0)
                message(FATAL_ERROR "Duktape configuration failed")
            endif()
    endif()
    
    add_library(duktape SHARED
    ${CMAKE_BINARY_DIR}/duktape/duk_config.h
    ${CMAKE_BINARY_DIR}/duktape/duktape.h
    ${CMAKE_BINARY_DIR}/duktape/duktape.c)
    set(DUKTAPE_LIBRARY duktape CACHE STRING "Duktape library target")
    set(DUKTAPE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/duktape CACHE PATH "Duktape include directory")

endfunction()
