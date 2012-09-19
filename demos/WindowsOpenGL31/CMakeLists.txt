SET(WGL_SOURCES WindowsOpenGL31.cpp  ../common/logo.c ../common/Vera.c ../common/glcommon.c)
SET(SDLGL_HEADERS ../common/logo.h ../common/Vera.h ../common/glcommon.h)
ADD_EXECUTABLE(WindowsOpenGL31 WIN32 ${WGL_SOURCES})
TARGET_LINK_LIBRARIES(WindowsOpenGL31 ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} OpenGL31Renderer aeongui)
