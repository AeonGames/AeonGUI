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

function(aeongui_configure_duktape)
    find_library(DUKTAPE_LIBRARY name duktape)
    if(DUKTAPE_LIBRARY)
        find_path(DUKTAPE_INCLUDE_DIR name duktape.h)
    else()
        find_package (Python2 COMPONENTS Interpreter Development)
        if(Python2_EXECUTABLE)
            file(WRITE ${CMAKE_BINARY_DIR}/find_pyyaml.py "import imp\ntry:\n\timp.find_module('yaml')\n\tprint 'PYYAML-FOUND'\nexcept ImportError:\n\tprint 'PYYAML-NOTFOUND'\n")
            execute_process ( COMMAND ${Python2_EXECUTABLE} ${CMAKE_BINARY_DIR}/find_pyyaml.py OUTPUT_VARIABLE PYYAML OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT PYYAML)
                message(STATUS "PYYAML module not found, attempting install")
                execute_process ( COMMAND ${Python2_EXECUTABLE} -m pip install pyyaml RESULT_VARIABLE PIP_INSTALL_RESULT)
                if(NOT PIP_INSTALL_RESULT EQUAL 0)
                    message(FATAL_ERROR "PYYAML instalation failed. Try manual instalation Interpreter found was: ${Python2_EXECUTABLE}")
                endif()
            endif()
        else()
            message(FATAL_ERROR "You need a Python 2 Interpreter installed in order to generate duktape code.")
        endif()
        download("https://duktape.org/duktape-2.3.0.tar.xz" "duktape-2.3.0.tar.xz")
        decompress("duktape-2.3.0.tar.xz" "duktape-2.3.0")
    endif()
endfunction(aeongui_configure_duktape)
