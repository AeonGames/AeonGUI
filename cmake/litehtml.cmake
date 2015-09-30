# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.

include(functions)
find_package(GIT REQUIRED)
if(GIT_FOUND)
	gitclone(https://github.com/Kwizatz/litehtml.git litehtml)
	subdirs(${CMAKE_SOURCE_DIR}/litehtml)
endif(GIT_FOUND)
