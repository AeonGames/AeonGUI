SET(WGL_SOURCES wgl.cpp  ../common/logo.c ../common/Vera.c ../common/glcommon.c)
SET(SDLGL_HEADERS ../common/logo.h ../common/Vera.h ../common/glcommon.h)
ADD_EXECUTABLE(wgl WIN32 ${WGL_SOURCES})
TARGET_LINK_LIBRARIES(wgl ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} OpenGLRenderer aeongui)
