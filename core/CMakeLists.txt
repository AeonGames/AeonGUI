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
Rect.cpp
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
set(AEONGUI_LIBRARIES pcx)
if(USE_PNG)
set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
endif(USE_PNG)

SOURCE_GROUP("Resource Files" FILES ${AEONGUI_RESOURCES})
add_library(AeonGUI ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES})
target_link_libraries(AeonGUI ${AEONGUI_LIBRARIES})
