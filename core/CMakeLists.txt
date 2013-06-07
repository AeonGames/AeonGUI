SET(AEONGUI_HEADERS
../include/KeyListener.h
../include/MouseListener.h
../include/Widget.h
../include/Button.h
../include/ScrollBar.h
../include/MainWindow.h
../include/Static.h
../include/Image.h
../include/Font.h
../include/Renderer.h
../include/Integer.h
../include/Rect.h
../include/Color.h
)
SET(AEONGUI_SOURCES
Renderer.cpp
Font.cpp
Widget.cpp
Button.cpp
ScrollBar.cpp
MainWindow.cpp
Image.cpp
Color.cpp
)
SET(AEONGUI_RESOURCES
resources/close.c
resources/maximize.c
resources/minimize.c
resources/restore.c
resources/scrolldown.c
resources/scrolldownpressed.c
resources/scrollup.c
resources/scrolluppressed.c
resources/scrollleft.c
resources/scrollleftpressed.c
resources/scrollright.c
resources/scrollrightpressed.c
resources/close.h
resources/maximize.h
resources/minimize.h
resources/restore.h
resources/scrolldown.h
resources/scrolldownpressed.h
resources/scrollup.h
resources/scrolluppressed.h
resources/scrollleft.h
resources/scrollleftpressed.h
resources/scrollright.h
resources/scrollrightpressed.h
resources/close_down.c
resources/maximize_down.c
resources/minimize_down.c
resources/restore_down.c
resources/close_down.h
resources/maximize_down.h
resources/minimize_down.h
resources/restore_down.h
)

IF(USE_HASH_CRC)
ADD_DEFINITIONS(-DUSE_HASH_CRC)
SET(AEONGUI_SOURCES ${AEONGUI_SOURCES} crc/crc.c)
SET(AEONGUI_HEADERS ${AEONGUI_HEADERS} crc/crc.h)
ENDIF(USE_HASH_CRC)
IF(USE_HASH_SHA1)
ADD_DEFINITIONS(-DUSE_HASH_SHA1)
SET(AEONGUI_SOURCES ${AEONGUI_SOURCES} sha1/sha1.cpp)
SET(AEONGUI_HEADERS ${AEONGUI_HEADERS} sha1/sha1.h)
ENDIF(USE_HASH_SHA1)
IF(USE_HASH_MD5)
ADD_DEFINITIONS(-DUSE_HASH_MD5)
SET(AEONGUI_SOURCES ${AEONGUI_SOURCES} md5/md5.c)
SET(AEONGUI_HEADERS ${AEONGUI_HEADERS} md5/md5.h)
ENDIF(USE_HASH_MD5)

SOURCE_GROUP("Resource Files" FILES ${AEONGUI_RESOURCES})
ADD_LIBRARY(AeonGUI ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES})
target_link_libraries(AeonGUI pcx)
