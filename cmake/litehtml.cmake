# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.

find_package(GIT REQUIRED)
function(gitclone repo path)
	if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${path}")
		message(STATUS "Cloning ${repo}, please wait")
		execute_process(COMMAND ${GIT_EXECUTABLE} clone ${repo} ${path} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE git_result OUTPUT_VARIABLE git_output ERROR_VARIABLE git_output)
		if(NOT git_result EQUAL 0)
			MESSAGE(FATAL_ERROR "Cloning ${repo} failed.\nResult: ${git_result}\nOutput: ${git_output}")
		endif(NOT git_result EQUAL 0)
	endif(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${path}")
endfunction(gitclone repo path)

if(GIT_FOUND)
	gitclone(https://github.com/Kwizatz/litehtml.git litehtml)
	subdirs(${CMAKE_SOURCE_DIR}/litehtml)
endif(GIT_FOUND)
