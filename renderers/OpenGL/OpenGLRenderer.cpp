/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

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
#define NOMINMAX
#include "OpenGLRenderer.h"
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <wingdi.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "glext.h"
#include "glcorearb.h"
#include "wglext.h"
#else
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include "glxext.h"
#include "glext.h"
#include "glcorearb.h"
#define GLGETPROCADDRESS(glFunction,glFunctionType) \
    glFunction = ( glFunctionType ) glXGetProcAddress ( (const GLubyte*) #glFunction )
#endif

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <wchar.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "OpenGL.h"
#include "Log.h"
#include "Font.h"
#include "fontstructs.h"

namespace AeonGUI
{
#include "vertex_shader.h"
#include "fragment_shader.h"

    static char log_buffer[1024] = {0};

    OpenGLRenderer::OpenGLRenderer()
    {
    }

    OpenGLRenderer::~OpenGLRenderer()
    {
    }

    bool OpenGLRenderer::Initialize ()
    {
        static bool functionsloaded = LoadOpenGLFunctions();
        if ( !functionsloaded )
        {
            return false;
        }
        // Compile Shaders
        char log_buffer[1024] = { 0 };
        GLint info_log_length;
        GLint shader_code_length;
        GLuint lVertexShader;
        GLuint lFragmentShader;
        if ( ( lVertexShader = glCreateShader ( GL_VERTEX_SHADER ) ) == 0 )
        {
            AEONGUI_OPENGL_CHECK_ERROR();
            return false;
        }
        if ( ( lFragmentShader = glCreateShader ( GL_FRAGMENT_SHADER ) ) == 0 )
        {
            AEONGUI_OPENGL_CHECK_ERROR();
            return false;
        }

        shader_code_length = static_cast<GLint> ( strlen ( vertex_shader ) );
        glShaderSource ( lVertexShader, 1, ( const char ** ) &vertex_shader, &shader_code_length );
        AEONGUI_OPENGL_CHECK_ERROR();
        shader_code_length = static_cast<GLint> ( strlen ( fragment_shader ) );
        glShaderSource ( lFragmentShader, 1, ( const char ** ) &fragment_shader, &shader_code_length );
        AEONGUI_OPENGL_CHECK_ERROR();


        glCompileShader ( lVertexShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        glCompileShader ( lFragmentShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        GLint compile_status;

        glGetShaderiv ( lVertexShader, GL_COMPILE_STATUS, &compile_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetShaderiv ( lVertexShader, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( lVertexShader, info_log_length, nullptr, log_buffer );
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }

        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        glGetShaderiv ( lFragmentShader, GL_COMPILE_STATUS, &compile_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetShaderiv ( lFragmentShader, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( lFragmentShader, info_log_length, nullptr, log_buffer );
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }
        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        // Create Shader Program
        mShaderProgram = glCreateProgram();
        AEONGUI_OPENGL_CHECK_ERROR();
        GLint link_status;

        glAttachShader ( mShaderProgram, lVertexShader );
        AEONGUI_OPENGL_CHECK_ERROR();
        glAttachShader ( mShaderProgram, lFragmentShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        glLinkProgram ( mShaderProgram );
        AEONGUI_OPENGL_CHECK_ERROR();

        // http://stackoverflow.com/questions/9113154/proper-way-to-delete-glsl-shader
        glDeleteShader ( lVertexShader );
        AEONGUI_OPENGL_CHECK_ERROR();
        glDeleteShader ( lFragmentShader );
        AEONGUI_OPENGL_CHECK_ERROR();

        glGetProgramiv ( mShaderProgram, GL_LINK_STATUS, &link_status );
        AEONGUI_OPENGL_CHECK_ERROR();
        glGetProgramiv ( mShaderProgram, GL_INFO_LOG_LENGTH, &info_log_length );
        AEONGUI_OPENGL_CHECK_ERROR();
        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetProgramInfoLog ( mShaderProgram, info_log_length, nullptr, log_buffer );
            AEONGUI_OPENGL_CHECK_ERROR();
            AEONGUI_LOG_ERROR ( "Error: %s\n", log_buffer );
        }
        if ( link_status != GL_TRUE )
        {
            return false;
        }

        // Generate VAO
        glGenVertexArrays ( 1, &mVertexArrayObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        // Generate VBO
        glGenBuffers ( 1, &mVertexBufferObject );
        AEONGUI_OPENGL_CHECK_ERROR();
        return true;
    }

    void OpenGLRenderer::Finalize()
    {
#if 0
        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
            screen_texture = 0;
        }
#endif
        // OpenGL Delete functions silently ignore invalid values
        glDeleteProgram ( mShaderProgram );
        AEONGUI_OPENGL_CHECK_ERROR();
        mShaderProgram = 0;

        glDeleteVertexArrays ( 1, &mVertexArrayObject );
        AEONGUI_OPENGL_CHECK_ERROR();
        mVertexArrayObject = 0;

        glDeleteBuffers ( 1, &mVertexBufferObject );
        AEONGUI_OPENGL_CHECK_ERROR();
        mVertexBufferObject = 0;
    }

    void OpenGLRenderer::BeginRender()
    {
    }

    void OpenGLRenderer::EndRender()
    {
        glUseProgram ( mShaderProgram );
        AEONGUI_OPENGL_CHECK_ERROR();

        glBindVertexArray ( mVertexArrayObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        glBindBuffer (  GL_ARRAY_BUFFER, mVertexBufferObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        AEONGUI_OPENGL_CHECK_ERROR();

        glEnable ( GL_BLEND );
        AEONGUI_OPENGL_CHECK_ERROR();

        glDepthMask ( GL_FALSE );
        glDisable ( GL_DEPTH_TEST );
#if 0
        glBindTexture ( GL_TEXTURE_2D, screen_texture );
        AEONGUI_OPENGL_CHECK_ERROR();

        glTexSubImage2D ( GL_TEXTURE_2D, 0, 0, 0, aWidth, aHeight, GL_BGRA, GL_UNSIGNED_BYTE, screen_bitmap );
        AEONGUI_OPENGL_CHECK_ERROR();

        glDrawArrays ( GL_TRIANGLE_STRIP, 0, 4 );
        AEONGUI_OPENGL_CHECK_ERROR();
#endif
    }

    uint32_t AeonGUI::OpenGLRenderer::SurfaceWidth() const
    {
        return 0;
    }

    uint32_t OpenGLRenderer::SurfaceHeight() const
    {
        return 0;
    }

    void * OpenGLRenderer::MapMemory()
    {
        return nullptr;
    }

    void OpenGLRenderer::UnmapMemory()
    {
    }

    void OpenGLRenderer::ReSize ( uint32_t aWidth, uint32_t aHeight )
    {
#if 0
        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
        }

        glGetIntegerv ( GL_MAX_TEXTURE_SIZE, &max_texture_size );
        AEONGUI_OPENGL_CHECK_ERROR();
        if ( ( aWidthidth > max_texture_size ) ||
             ( aHeighteight > max_texture_size ) )
        {
            AEONGUI_LOG_ERROR ( "Error: %s\n", "Screen texture dimensions surpass maximum allowed OpenGL texture size" );
        }

        glGenTextures ( 1, &screen_texture );
        AEONGUI_OPENGL_CHECK_ERROR();
        glBindTexture ( GL_TEXTURE_2D, screen_texture );
        AEONGUI_OPENGL_CHECK_ERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        AEONGUI_OPENGL_CHECK_ERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        AEONGUI_OPENGL_CHECK_ERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        AEONGUI_OPENGL_CHECK_ERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        AEONGUI_OPENGL_CHECK_ERROR();
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8, aWidthidth, aHeighteight, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL );
        AEONGUI_OPENGL_CHECK_ERROR();
#endif
        const float pixel_offset = -0.375f;
        float width = float ( aWidth ) + pixel_offset;
        float height = float ( aHeight ) + pixel_offset;

        // glOrtho
        // left          right  bottom  top           near  far
        // pixel_offset, width, height, pixel_offset, 0.0f, 1.0f
        // X
        GLfloat projection[16];
        projection[0] = 2.0f / ( width - pixel_offset );
        projection[1] = 0.0f;
        projection[2] = 0.0f;
        projection[3] = 0.0f;

        // Y
        projection[4] = 0.0f;
        projection[5] = 2.0f / ( pixel_offset - height );
        projection[6] = 0.0f;
        projection[7] = 0.0f;

        // Z
        projection[8] = 0.0f;
        projection[9] = 0.0f;
        projection[10] = -2.0f;
        projection[11] = 0.0f;

        // Pos
        projection[12] = - ( ( width + pixel_offset ) / ( width - pixel_offset ) );
        projection[13] = - ( ( pixel_offset + height ) / ( pixel_offset - height ) );
        projection[14] = -1.0f;
        projection[15] = 1.0f;

        glUseProgram ( mShaderProgram );
        AEONGUI_OPENGL_CHECK_ERROR();

        GLint projection_matrix = glGetUniformLocation ( mShaderProgram, "projection_matrix" );
        AEONGUI_OPENGL_CHECK_ERROR();
        if ( projection_matrix > -1 )
        {
            glUniformMatrix4fv ( projection_matrix, 1, GL_FALSE, projection );
            AEONGUI_OPENGL_CHECK_ERROR();
        }

        GLint screen_texture_index = glGetUniformLocation ( mShaderProgram, "screen_texture" );
        AEONGUI_OPENGL_CHECK_ERROR();
        if ( screen_texture_index > -1 )
        {
            glUniform1i ( screen_texture_index, 0 );
            AEONGUI_OPENGL_CHECK_ERROR();
        }

        // Create VBO
        GLfloat vertices[16] =
        {
            /* position */ 0.0f, 0.0f,                                                                /* uv */ 0.0f, 0.0f,
            /* position */ static_cast<float> ( aWidth ), 0.0f,                                 /* uv */ 1.0f, 0.0f,
            /* position */ 0.0f, static_cast<float> ( aHeight ),                                /* uv */ 0.0f, 1.0f,
            /* position */ static_cast<float> ( aWidth ), static_cast<float> ( aHeight ), /* uv */ 1.0f, 1.0f
        };


        glBindVertexArray ( mVertexArrayObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        glBindBuffer ( GL_ARRAY_BUFFER, mVertexBufferObject );
        AEONGUI_OPENGL_CHECK_ERROR();

        glBufferData ( GL_ARRAY_BUFFER, sizeof ( GLfloat ) * 16, &vertices[0], GL_STATIC_DRAW );
        AEONGUI_OPENGL_CHECK_ERROR();

        GLint position = glGetAttribLocation ( mShaderProgram, "position" );
        AEONGUI_OPENGL_CHECK_ERROR();
        glVertexAttribPointer ( position, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, ( ( void* ) 0 ) );
        AEONGUI_OPENGL_CHECK_ERROR();
        glEnableVertexAttribArray ( position );
        AEONGUI_OPENGL_CHECK_ERROR();

        GLint uv = glGetAttribLocation ( mShaderProgram, "uv" );
        glVertexAttribPointer ( uv, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, ( ( void* ) ( sizeof ( float ) * 2 ) ) );
        AEONGUI_OPENGL_CHECK_ERROR();
        glEnableVertexAttribArray ( uv );
        AEONGUI_OPENGL_CHECK_ERROR();
    }

}
