include_directories(${CAIRO_INCLUDE_DIRS})

# Locate GL extension headers from vcpkg / system, download as fallback.
# macOS uses Apple's own <OpenGL/gl3.h> framework header instead, so these
# Khronos headers are neither needed nor included there.
if(NOT APPLE)
    find_path(GLCOREARB_INCLUDE_DIR GL/glcorearb.h)
    if(NOT GLCOREARB_INCLUDE_DIR)
        message(STATUS "Downloading KHR/khrplatform.h")
        file(DOWNLOAD https://registry.khronos.org/EGL/api/KHR/khrplatform.h
                    "${CMAKE_CURRENT_BINARY_DIR}/KHR/khrplatform.h"
            SHOW_PROGRESS
            STATUS _khr_status)
        message(STATUS "Downloading GL/glcorearb.h")
        file(DOWNLOAD https://registry.khronos.org/OpenGL/api/GL/glcorearb.h
                    "${CMAKE_CURRENT_BINARY_DIR}/GL/glcorearb.h"
            SHOW_PROGRESS
            STATUS _glcorearb_status)
        list(GET _khr_status 0 _khr_code)
        list(GET _glcorearb_status 0 _glcorearb_code)
        if(NOT _khr_code EQUAL 0 OR NOT _glcorearb_code EQUAL 0)
            message(FATAL_ERROR "Failed to download GL extension headers: khrplatform=${_khr_status} glcorearb=${_glcorearb_status}")
        endif()
        set(GLCOREARB_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "GL extension headers" FORCE)
    endif()
    include_directories(${GLCOREARB_INCLUDE_DIR})
endif()

if(WIN32)
    find_path(WGLEXT_INCLUDE_DIR GL/wglext.h)
    if(NOT WGLEXT_INCLUDE_DIR)
        message(STATUS "Downloading GL/wglext.h")
        file(DOWNLOAD https://registry.khronos.org/OpenGL/api/GL/wglext.h
                    "${CMAKE_CURRENT_BINARY_DIR}/GL/wglext.h"
            SHOW_PROGRESS
            STATUS _wglext_status)
        list(GET _wglext_status 0 _wglext_code)
        if(NOT _wglext_code EQUAL 0)
            message(FATAL_ERROR "Failed to download GL/wglext.h: ${_wglext_status}")
        endif()
        set(WGLEXT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "WGL extension headers" FORCE)
    endif()
    include_directories(${WGLEXT_INCLUDE_DIR})
    add_executable(OpenGLDemo Common.h WindowsOpenGL.cpp)
elseif(APPLE)
    add_executable(OpenGLDemo Common.h MacOSOpenGL.mm)
    set_source_files_properties(MacOSOpenGL.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
    set_target_properties(OpenGLDemo PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
else()
    find_package(X11)
    add_executable(OpenGLDemo Common.h LinuxOpenGL.cpp)
endif()

#set_target_properties(OpenGLDemo PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "file:///${PROJECT_SOURCE_DIR}/images/tiger-style.svg")
set_target_properties(OpenGLDemo PROPERTIES
    AEONGUI_SVG_FILES "${PROJECT_SOURCE_DIR}/images/text.svg;${PROJECT_SOURCE_DIR}/images/textpath.svg;${PROJECT_SOURCE_DIR}/images/drawing.svg;${PROJECT_SOURCE_DIR}/images/tiger-style.svg;${PROJECT_SOURCE_DIR}/images/gradient.svg;${PROJECT_SOURCE_DIR}/images/curve.svg;${PROJECT_SOURCE_DIR}/images/arc.svg;${PROJECT_SOURCE_DIR}/images/line.svg;${PROJECT_SOURCE_DIR}/images/rasterimage.svg;${PROJECT_SOURCE_DIR}/images/button.svg;${PROJECT_SOURCE_DIR}/images/hover-demo.svg;${PROJECT_SOURCE_DIR}/images/smil-demo.svg;${PROJECT_SOURCE_DIR}/images/fps.xhtml"
)
set_target_properties(OpenGLDemo PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")

if(APPLE)
    target_link_libraries(OpenGLDemo AeonGUI OpenGL::GL "-framework Cocoa" "-framework OpenGL")
else()
    target_link_libraries(OpenGLDemo AeonGUI ${X11_LIBRARIES} OpenGL::GL)
endif()

# Compile images/fps.xhtml into a C++ document the demo can host directly
# (xmlcxx). Run OpenGLDemo with the literal argument "compiled-fps" to load
# the generated FpsDocument instead of parsing an SVG/XHTML file at runtime.
if(TARGET xmlcxx)
    include(xmlcxx)
    xmlcxx_generate(FPS_DOCUMENT_SRCS
        "${PROJECT_SOURCE_DIR}/images/fps.xhtml"
        CLASS FpsDocument)
    target_sources(OpenGLDemo PRIVATE ${FPS_DOCUMENT_SRCS})
    target_include_directories(OpenGLDemo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(OpenGLDemo PRIVATE AEONGUI_HAVE_COMPILED_FPS)
endif()

if(MINGW)
    fix_compile_commands(OpenGLDemo)
endif()
