SET(FONT2C_HEADERS ../../common/fontstructs.h ../../include/Integer.h)
SET(FONT2C_SOURCES font2c.cpp)

include_directories(${FREETYPE_INCLUDE_DIR_freetype2} ${FREETYPE_INCLUDE_DIR_ft2build})

add_executable(font2c ${FONT2C_HEADERS} ${FONT2C_SOURCES} )

if(NOT FREETYPE_FOUND)
add_dependencies(font2c ${FREETYPE_LIBRARIES})
endif(NOT FREETYPE_FOUND)
TARGET_LINK_LIBRARIES(font2c ${FREETYPE_LIBRARIES} pcx)

