# Helper static library for PCX file reading and writting.
set(PCX_SOURCES pcx.cpp)
set(PCX_HEADERS pcx.h)
add_library(pcx STATIC ${PCX_SOURCES} ${PCX_HEADERS})
