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

add_executable(OpenGLDemo WindowsOpenGL.cpp)
set_target_properties(OpenGLDemo PROPERTIES DEBUG_ARGUMENTS "${CMAKE_SOURCE_DIR}/images/arc.svg")
target_link_libraries(OpenGLDemo AeonGUI ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
