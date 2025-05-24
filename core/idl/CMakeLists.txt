find_program(NPM_EXECUTABLE npm)
find_package (Python3 COMPONENTS Interpreter)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/node_modules/.bin/reffy AND NPM_EXECUTABLE)
    execute_process(COMMAND ${NPM_EXECUTABLE} install reffy)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/node_modules/.bin/reffy AND NPM_EXECUTABLE)
    if(MSYS)
        set(REFFY_EXECUTABLE bash ${CMAKE_SOURCE_DIR}/node_modules/.bin/reffy)
    else()
        set(REFFY_EXECUTABLE ${CMAKE_SOURCE_DIR}/node_modules/.bin/reffy)
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/node_modules/reffy/src/cli/parse-webidl.js AND NPM_EXECUTABLE)
    set(REFFY_PARSE_WEBIDL ${CMAKE_SOURCE_DIR}/node_modules/reffy/src/cli/parse-webidl.js)
endif()

if(REFFY_EXECUTABLE)
    add_custom_target(generate_idl_files
        COMMAND ${REFFY_EXECUTABLE} --spec https://www.w3.org/TR/SVG2/ --spec https://dom.spec.whatwg.org/ --module idl --output ..
        COMMAND ${CMAKE_COMMAND} -E copy SVG.idl ${CMAKE_CURRENT_SOURCE_DIR}/SVG.idl
        COMMAND ${CMAKE_COMMAND} -E copy dom.idl ${CMAKE_CURRENT_SOURCE_DIR}/dom.idl
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating idl files, please wait...")
    message(STATUS "The target generate_idl_files is now available")
endif()

if(REFFY_PARSE_WEBIDL)
    add_custom_command(OUTPUT SVG.json
        COMMAND node ${REFFY_PARSE_WEBIDL} SVG.idl > SVG.json
        DEPENDS SVG.idl
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Parsing SVG.idl")
    add_custom_command(OUTPUT dom.json
        COMMAND node ${REFFY_PARSE_WEBIDL} dom.idl > dom.json
        DEPENDS dom.idl
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Parsing dom.idl")
endif()

if(Python3_Interpreter_FOUND)

#function(generate_header interface)
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/include/aeongui/dom/Node.h
#                   COMMAND ${Python3_EXECUTABLE} generate_cxx_api.py dom.json "${CMAKE_SOURCE_DIR}/include/aeongui/dom" --interface ${interface}
#                    DEPENDS dom.json
#                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                    COMMENT "Generating ${interface}.h")
#endfunction()
#generate_source(Node)



    add_custom_target(generate_cxx_dom_api
        COMMAND ${Python3_EXECUTABLE} generate_cxx_api.py SVG.json "${CMAKE_SOURCE_DIR}/include/aeongui/dom" --interface SVGElement
        COMMAND ${Python3_EXECUTABLE} generate_cxx_api.py dom.json "${CMAKE_SOURCE_DIR}/include/aeongui/dom" --interface Node
        DEPENDS SVG.json dom.json
        SOURCES SVG.json dom.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating C++ API from idl")
        message(STATUS "The target generate_cxx_dom_api is now available")
endif()
