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

include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_REQUIRED_QUIET OFF)
set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})

check_include_file_cxx("GL/glcorearb.h" HAS_GLCOREARB_H)
if(NOT HAS_GLCOREARB_H)
    message(STATUS "Downloading KHR/khrplatform.h")
    file(DOWNLOAD https://www.khronos.org/registry/EGL/api/KHR/khrplatform.h
                "${CMAKE_CURRENT_BINARY_DIR}/KHR/khrplatform.h"
        SHOW_PROGRESS)
    message(STATUS "Downloading GL/glcorearb.h")
    file(DOWNLOAD https://www.khronos.org/registry/OpenGL/api/GL/glcorearb.h
                "${CMAKE_CURRENT_BINARY_DIR}/GL/glcorearb.h"
        SHOW_PROGRESS)
    set(HAS_GLCOREARB_H 1 CACHE INTERNAL "Downloaded header file" FORCE)
endif()

if(WIN32)
    check_include_files("GL/gl.h;GL/wglext.h" HAS_WGLEXT_H)
    if(NOT HAS_WGLEXT_H)
    message(STATUS "Downloading GL/wglext.h")
    file(DOWNLOAD https://www.khronos.org/registry/OpenGL/api/GL/wglext.h
                    "${CMAKE_CURRENT_BINARY_DIR}/GL/wglext.h"
            SHOW_PROGRESS)
    set(HAS_WGLEXT_H 1 CACHE INTERNAL "Downloaded header file" FORCE)
    endif()
endif()

add_executable(OpenGLDemo ${WGL_SOURCES} ${WGL_HEADERS} ${WGL_DEPENDENCIES})
set_target_properties(OpenGLDemo PROPERTIES DEBUG_ARGUMENTS "${CMAKE_SOURCE_DIR}/images/tiger.svg")
target_link_libraries(OpenGLDemo AeonGUI ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
