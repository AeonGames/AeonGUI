SET(OPENGL_RENDERER_SOURCES OpenGLRenderer.cpp)
SET(OPENGL_RENDERER_HEADERS 
    ../../include/Renderer.h 
    ../../include/OpenGLRenderer.h 
    ../../common/fontstructs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vertex_shader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fragment_shader.h
    )

add_custom_command( OUTPUT 
                        ${CMAKE_CURRENT_SOURCE_DIR}/vertex_shader.h
                    COMMAND ./convert-shaders ${CMAKE_CURRENT_SOURCE_DIR}/vertex_shader
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/vertex_shader
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating Vertex Shader Arrays" VERBATIM)
                    
add_custom_command( OUTPUT 
                        ${CMAKE_CURRENT_SOURCE_DIR}/fragment_shader.h
                    COMMAND ./convert-shaders ${CMAKE_CURRENT_SOURCE_DIR}/fragment_shader
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/fragment_shader
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating Fragment Shader Arrays" VERBATIM)

add_library(OpenGLRenderer SHARED ${OPENGL_RENDERER_HEADERS} ${OPENGL_RENDERER_SOURCES})
add_dependencies(OpenGLRenderer AeonGUI)
if(WIN32)
TARGET_LINK_LIBRARIES(OpenGLRenderer AeonGUI opengl32 glu32)
IF(MSVC)
SET_TARGET_PROPERTIES(OpenGLRenderer PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} /DNOMINMAX /DDLL_EXPORT /D_CRT_SECURE_NO_WARNINGS")
ENDIF(MSVC)
IF(MSYS)
SET_TARGET_PROPERTIES(OpenGLRenderer PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -DNOMINMAX -DDLL_EXPORT -DWIN32")
ENDIF(MSYS)
endif(WIN32)