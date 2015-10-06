# Copyright 2015 AeonGames, Rodrigo Hernandez
# Licensed under the terms of the Apache 2.0 License.

include(functions)
download("http://grail.cba.csuohio.edu/~somos/xxd-1.10.tar.gz" "xxd-1.10.tar.gz")
decompress("xxd-1.10.tar.gz" "xxd-1.10")
add_executable(xxd ${CMAKE_SOURCE_DIR}/xxd-1.10/xxd.c)
