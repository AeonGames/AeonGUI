set(OPENGL_RENDERER_SOURCES 
	OpenGL.cpp
	OpenGLContext.cpp
	OpenGLRenderer.cpp
)
set(OPENGL_RENDERER_HEADERS
    ../../include/Renderer.h 
    ../../include/OpenGLRenderer.h 
    ../../common/fontstructs.h
    ../../common/Log.h
	OpenGL.h
    vertex_shader.h
    fragment_shader.h
    )

add_library(OpenGLRenderer SHARED ${OPENGL_RENDERER_HEADERS} ${OPENGL_RENDERER_SOURCES})
add_dependencies(OpenGLRenderer AeonGUI)
target_link_libraries(OpenGLRenderer AeonGUI ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
if(WIN32)
	set(CONVERT_SHADERS convert-shaders.bat)
	set_target_properties(OpenGLRenderer PROPERTIES COMPILE_FLAGS "-DDLL_EXPORT -DWIN32 -D_CRT_SECURE_NO_WARNINGS")
endif(WIN32)
if(UNIX)
	set(CONVERT_SHADERS convert-shaders)
endif(UNIX)

add_custom_command( OUTPUT vertex_shader.h fragment_shader.h
                    COMMAND ${CONVERT_SHADERS} ARGS vertex_shader
                    COMMAND ${CONVERT_SHADERS} ARGS fragment_shader
                    DEPENDS vertex_shader fragment_shader
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating Shader Arrays" VERBATIM)
