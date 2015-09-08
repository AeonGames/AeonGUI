SET(WGL_SOURCES WindowsOpenGL.cpp  ../common/logo.c ../common/Vera.c ../common/glcommon.c)
SET(WGL_HEADERS ../common/logo.h ../common/Vera.h ../common/glcommon.h)
SET(WGL_DEPENDENCIES 
    ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.pcx 
    ${CMAKE_CURRENT_BINARY_DIR}/ScaleTest.pcx 
    ${CMAKE_CURRENT_BINARY_DIR}/Patch9Test.pcx
    ${CMAKE_CURRENT_BINARY_DIR}/WindowFrame.pcx
    )

if(USE_PNG)
SET(WGL_DEPENDENCIES ${WGL_DEPENDENCIES} ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.png)
add_custom_command(OUTPUT 
                        ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.png
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/AeonGUILogoBlBkg.png ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.png
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/AeonGUILogoBlBkg.png
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying AeonGUILogoBlBkg.png" VERBATIM)
add_custom_command(OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/WindowFrame.png
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/WindowFrame.png ${CMAKE_CURRENT_BINARY_DIR}/WindowFrame.png
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/WindowFrame.png
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying WindowFrame.png" VERBATIM)                    
endif(USE_PNG)
add_custom_command(OUTPUT 
                        ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.pcx
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/AeonGUILogoBlBkg.pcx ${CMAKE_CURRENT_BINARY_DIR}/AeonGUILogoBlBkg.pcx
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/AeonGUILogoBlBkg.pcx
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying AeonGUILogoBlBkg.pcx" VERBATIM)

add_custom_command(OUTPUT 
                        ${CMAKE_CURRENT_BINARY_DIR}/ScaleTest.pcx
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/ScaleTest.pcx ${CMAKE_CURRENT_BINARY_DIR}/ScaleTest.pcx
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/ScaleTest.pcx
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying ScaleTest.pcx" VERBATIM)

add_custom_command(OUTPUT 
                        ${CMAKE_CURRENT_BINARY_DIR}/Patch9Test.pcx
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/Patch9Test.pcx ${CMAKE_CURRENT_BINARY_DIR}/Patch9Test.pcx
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/Patch9Test.pcx
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying Patch9Test.pcx" VERBATIM)
                    
add_custom_command(OUTPUT
                        ${CMAKE_CURRENT_BINARY_DIR}/WindowFrame.pcx
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../common/WindowFrame.pcx ${CMAKE_CURRENT_BINARY_DIR}/WindowFrame.pcx
                    DEPENDS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/../common/WindowFrame.pcx
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Copying WindowFrame.pcx" VERBATIM)


ADD_EXECUTABLE(WindowsOpenGL ${WGL_SOURCES} ${WGL_HEADERS} ${WGL_DEPENDENCIES})
TARGET_LINK_LIBRARIES(WindowsOpenGL ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} OpenGLRenderer AeonGUI)
