SET(WGL_SOURCES WindowsOpenGL.cpp  ../common/logo.c ../common/Vera.c ../common/glcommon.c)
SET(WGL_HEADERS ../common/logo.h ../common/Vera.h ../common/glcommon.h)
ADD_EXECUTABLE(WindowsOpenGL ${WGL_SOURCES} ${WGL_HEADERS})
TARGET_LINK_LIBRARIES(WindowsOpenGL ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} OpenGLRenderer aeongui)
