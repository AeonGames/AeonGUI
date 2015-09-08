SET(OPENGL_RENDERER_SOURCES OpenGLRenderer.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(OPENGL_RENDERER_HEADERS
    ../../include/Renderer.h 
    ../../include/OpenGLRenderer.h 
    ../../common/fontstructs.h
    vertex_shader.h
    fragment_shader.h
    )

add_custom_command( OUTPUT vertex_shader.h fragment_shader.h
                    COMMAND convert-shaders vertex_shader
                    COMMAND convert-shaders fragment_shader
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating Shader Arrays" VERBATIM)

add_library(OpenGLRenderer SHARED ${OPENGL_RENDERER_HEADERS} ${OPENGL_RENDERER_SOURCES})
add_dependencies(OpenGLRenderer AeonGUI)
target_link_libraries(OpenGLRenderer AeonGUI ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
if(WIN32)
	set_target_properties(OpenGLRenderer PROPERTIES COMPILE_FLAGS "-DNOMINMAX -DDLL_EXPORT -DWIN32 -D_CRT_SECURE_NO_WARNINGS")
endif(WIN32)
