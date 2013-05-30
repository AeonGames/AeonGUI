SET(LINUXOPENGL_SRCS LinuxOpenGL.cpp ../common/logo.c ../common/Vera.c ../common/glcommon.c)
ADD_EXECUTABLE(LinuxOpenGL ${LINUXOPENGL_SRCS})

SET_TARGET_PROPERTIES(LinuxOpenGL PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -Wall -g")

ADD_DEPENDENCIES(LinuxOpenGL OpenGLRenderer AeonGUI)
TARGET_LINK_LIBRARIES(LinuxOpenGL OpenGLRenderer AeonGUI ${OPENGL_LIBRARIES} X11 Xxf86vm rt)

