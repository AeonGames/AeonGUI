/******************************************************************************
Copyright 2015 Rodrigo Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************/
#include <cstdint>
#include <algorithm>
#include "OpenGL.h"
#include "Log.h"

#ifdef WIN32
#include "wglext.h"
#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
        glFunction = (glFunctionType)wglGetProcAddress(#glFunction); \
        if (glFunction == nullptr) { AEONGUI_LOG_ERROR("OpenGL: Unable to load %s function.", #glFunction); if(!contextAvailable){DestroyOpenGLContext();} return false; }}
#else
#include <GL/gl.h>
#include <GL/glx.h>
#include "glxext.h"
#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
    glFunction = ( glFunctionType ) glXGetProcAddress ( (const GLubyte*) #glFunction ); \
    if (glFunction==nullptr) { AEONGUI_LOG_ERROR("%s:%d OpenGL: Unable to load %s.",__FUNCTION__,__LINE__,#glFunction);return false;}}
#endif


namespace AeonGUI
{

#include "vertex_shader.h"
#include "fragment_shader.h"

    PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
    PFNGLCREATESHADERPROC glCreateShader = nullptr;
    PFNGLGENVERTEXARRAYSPROC  glGenVertexArrays = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
    PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
    PFNGLATTACHSHADERPROC glAttachShader = nullptr;
    PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
    PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
    PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
    PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
    PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;

    uint32_t gVertexShader;
    uint32_t gFragmentShader;
    uint32_t gShaderProgram;
    uint32_t gVertexBufferObject;
    uint32_t gVertexArrayObject;

    static bool InitializeShaderProgram()
    {
        // Compile Shaders
        char log_buffer[1024] = { 0 };
        GLint info_log_length;
        GLint shader_code_length;
        if ( ( gVertexShader = glCreateShader ( GL_VERTEX_SHADER ) ) == 0 )
        {
            AEONGUI_OPENGL_CHECK_ERROR();
            return false;
        }
        if ( ( gFragmentShader = glCreateShader ( GL_FRAGMENT_SHADER ) ) == 0 )
        {
            AEONGUI_OPENGL_CHECK_ERROR();
            return false;
        }

        shader_code_length = static_cast<GLint> ( strlen ( vertex_shader ) );
        glShaderSource ( gVertexShader, 1, ( const char ** ) &vertex_shader, &shader_code_length );
        AEONGUI_OPENGL_CHECK_ERROR();
        shader_code_length = static_cast<GLint> ( strlen ( fragment_shader ) );
        glShaderSource ( gFragmentShader, 1, ( const char ** ) &fragment_shader, &shader_code_length );
        AEONGUI_OPENGL_CHECK_ERROR();


        glCompileShader ( gVertexShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        glCompileShader ( gFragmentShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        GLint compile_status;

        glGetShaderiv ( gVertexShader, GL_COMPILE_STATUS, &compile_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetShaderiv ( gVertexShader, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( gVertexShader, info_log_length, nullptr, log_buffer );
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }

        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        glGetShaderiv ( gFragmentShader, GL_COMPILE_STATUS, &compile_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetShaderiv ( gFragmentShader, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( gFragmentShader, info_log_length, nullptr, log_buffer );
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }
        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        // Create Shader Program
        gShaderProgram = glCreateProgram();
        AEONGUI_OPENGL_CHECK_ERROR();
        GLint link_status;

        glAttachShader ( gShaderProgram, gVertexShader );
        AEONGUI_OPENGL_CHECK_ERROR();
        glAttachShader ( gShaderProgram, gFragmentShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        glLinkProgram ( gShaderProgram );
        AEONGUI_OPENGL_CHECK_ERROR();

        glGetProgramiv ( gShaderProgram, GL_LINK_STATUS, &link_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetProgramiv ( gShaderProgram, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();
        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetProgramInfoLog ( gShaderProgram, info_log_length, nullptr, log_buffer );
            AEONGUI_OPENGL_CHECK_ERROR();
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }
        if ( link_status != GL_TRUE )
        {
            return false;
        }

        // Generate VAO
        glGenVertexArrays ( 1, &gVertexArrayObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        // Generate VBO
        glGenBuffers ( 1, &gVertexBufferObject );
        AEONGUI_OPENGL_CHECK_ERROR();
        return true;
    }

    bool AeonGUI::InitializeOpenGL()
    {
#ifdef WIN32
        const bool contextAvailable = ( wglGetCurrentContext() != nullptr );
#else
        const bool contextAvailable = ( glXGetCurrentContext() != nullptr );
#endif
        if ( !contextAvailable && !CreateOpenGLContext() )
        {
            return false;
        }

        GLGETPROCADDRESS ( PFNGLGENBUFFERSPROC, glGenBuffers );
        GLGETPROCADDRESS ( PFNGLCREATESHADERPROC, glCreateShader );
        GLGETPROCADDRESS ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
        GLGETPROCADDRESS ( PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog );
        GLGETPROCADDRESS ( PFNGLSHADERSOURCEPROC, glShaderSource );
        GLGETPROCADDRESS ( PFNGLATTACHSHADERPROC, glAttachShader );
        GLGETPROCADDRESS ( PFNGLCOMPILESHADERPROC, glCompileShader );
        GLGETPROCADDRESS ( PFNGLGETPROGRAMIVPROC, glGetProgramiv );
        GLGETPROCADDRESS ( PFNGLGETSHADERIVPROC, glGetShaderiv );
        GLGETPROCADDRESS ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
        GLGETPROCADDRESS ( PFNGLCREATEPROGRAMPROC, glCreateProgram );

        if ( !contextAvailable )
        {
            DestroyOpenGLContext();
        }
        return true;
    }

    void AeonGUI::FinalizeOpenGL()
    {
    }
}