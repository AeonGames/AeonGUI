include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CAIRO_INCLUDE_DIRS})
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
    add_executable(OpenGLDemo Common.h WindowsOpenGL.cpp)
else()
    find_package(X11)
    add_executable(OpenGLDemo Common.h LinuxOpenGL.cpp)
endif()

set_target_properties(OpenGLDemo PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "${CMAKE_SOURCE_DIR}/images/tiger-attrib.svg")
set_target_properties(OpenGLDemo PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
target_link_libraries(OpenGLDemo AeonGUI ${X11_LIBRARIES} OpenGL::GL)
if(MINGW)
    fix_compile_commands(OpenGLDemo)
endif()
configure_vscode(${CMAKE_SOURCE_DIR}/demos/OpenGL)
