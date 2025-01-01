find_program(REFFY_EXECUTABLE reffy HINTS ENV PATH_SUFFIXES /usr/bin /usr/local/bin ${CMAKE_SOURCE_DIR}/node_modules/.bin)
find_program(NPM_EXECUTABLE npm)
find_package (Python3 COMPONENTS Interpreter)

if(NOT REFFY_EXECUTABLE AND NPM_EXECUTABLE)
    message(STATUS "reffy not found, trying to install it")
    execute_process(COMMAND ${NPM_EXECUTABLE} install reffy)
    find_program(REFFY_EXECUTABLE reffy HINTS ENV PATH_SUFFIXES /usr/bin /usr/local/bin ${CMAKE_SOURCE_DIR}/node_modules/.bin)
elseif(NOT REFFY_EXECUTABLE AND NOT NPM_EXECUTABLE)
    message(WARNING "NodeJS not found, will use pre-generated idl files")
    message(WARNING "See ${CMAKE_CURRENT_SOURCE_DIR}/README.md for more information")
endif()

if(REFFY_EXECUTABLE)
    message(STATUS "Generating idl files, please wait...")
    execute_process(COMMAND bash ${REFFY_EXECUTABLE} --spec https://www.w3.org/TR/SVG2/ --module idl
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/SVG2.idl)
    string(REPLACE ".bin/reffy" "reffy/src/cli/parse-webidl.js" REFFY_PARSE_WEBIDL ${REFFY_EXECUTABLE})
    execute_process(COMMAND node ${REFFY_PARSE_WEBIDL} SVG2.idl
        OUTPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/SVG2.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(Python3_Interpreter_FOUND)
    message(STATUS "Found Python3, changes in idl files will be reflected in generated code")
endif()
