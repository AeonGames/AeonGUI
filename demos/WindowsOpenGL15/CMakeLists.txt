SET(WGL_SOURCES WindowsOpenGL15.cpp  ../common/logo.c ../common/Vera.c ../common/glcommon.c)
SET(SDLGL_HEADERS ../common/logo.h ../common/Vera.h ../common/glcommon.h)
ADD_EXECUTABLE(WindowsOpenGL15 WIN32 ${WGL_SOURCES})
TARGET_LINK_LIBRARIES(WindowsOpenGL15 ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} OpenGL15Renderer aeongui)
