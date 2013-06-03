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
#define GLGETPROCADDRESS(glFunction,glFunctionType) \
    glFunction = ( glFunctionType ) wglGetProcAddress ( #glFunction )
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
#define GLDECLAREFUNCTION(glFunction,glFunctionType) \
    static glFunctionType glFunction = NULL


#include <cassert>
#include <cstdlib>
#include <cwchar>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "Font.h"
#include "fontstructs.h"

#ifdef min
#undef min
#endif

///\todo Add preprocessor code to disable LOGERROR on Release Builds.

#define LOGERROR()  { \
                        GLenum error = glGetError(); \
                        if((error != 0)) \
                        { \
                            std::cout << "Error " << error << ": " << gluErrorString(error) <<\
                            " at function " << __FUNCTION__ << " at line " <<\
                            ", file " << __FILE__ << " at line " <<\
                            __LINE__ << std::endl; \
                        }\
                    }

namespace AeonGUI
{

#include "vertex_shader.h"
#include "fragment_shader.h"

    GLDECLAREFUNCTION ( glAttachShader,             PFNGLATTACHSHADERPROC            );
    GLDECLAREFUNCTION ( glCompileShader,            PFNGLCOMPILESHADERPROC           );
    GLDECLAREFUNCTION ( glCreateProgram,            PFNGLCREATEPROGRAMPROC           );
    GLDECLAREFUNCTION ( glCreateShader,             PFNGLCREATESHADERPROC            );
    GLDECLAREFUNCTION ( glDeleteProgram,            PFNGLDELETEPROGRAMPROC           );
    GLDECLAREFUNCTION ( glDeleteShader,             PFNGLDELETESHADERPROC            );
    GLDECLAREFUNCTION ( glEnableVertexAttribArray,  PFNGLENABLEVERTEXATTRIBARRAYPROC );
    GLDECLAREFUNCTION ( glGetAttribLocation,        PFNGLGETATTRIBLOCATIONPROC       );
    GLDECLAREFUNCTION ( glGetProgramiv,             PFNGLGETPROGRAMIVPROC            );
    GLDECLAREFUNCTION ( glGetProgramInfoLog,        PFNGLGETPROGRAMINFOLOGPROC       );
    GLDECLAREFUNCTION ( glGetShaderiv,              PFNGLGETSHADERIVPROC             );
    GLDECLAREFUNCTION ( glGetShaderInfoLog,         PFNGLGETSHADERINFOLOGPROC        );
    GLDECLAREFUNCTION ( glGetUniformLocation,       PFNGLGETUNIFORMLOCATIONPROC      );
    GLDECLAREFUNCTION ( glLinkProgram,              PFNGLLINKPROGRAMPROC             );
    GLDECLAREFUNCTION ( glShaderSource,             PFNGLSHADERSOURCEPROC            );
    GLDECLAREFUNCTION ( glUseProgram,               PFNGLUSEPROGRAMPROC              );
    GLDECLAREFUNCTION ( glUniform1i,                PFNGLUNIFORM1IPROC               );
    GLDECLAREFUNCTION ( glUniformMatrix4fv,         PFNGLUNIFORMMATRIX4FVPROC        );
    GLDECLAREFUNCTION ( glVertexAttribPointer,      PFNGLVERTEXATTRIBPOINTERPROC     );
    GLDECLAREFUNCTION ( glGenBuffers,               PFNGLGENBUFFERSPROC              );
    GLDECLAREFUNCTION ( glBindBuffer,               PFNGLBINDBUFFERPROC              );
    GLDECLAREFUNCTION ( glBufferData,               PFNGLBUFFERDATAPROC              );
    GLDECLAREFUNCTION ( glDeleteBuffers,            PFNGLDELETEBUFFERSPROC           );
    GLDECLAREFUNCTION ( glGenVertexArrays,          PFNGLGENVERTEXARRAYSPROC         );
    GLDECLAREFUNCTION ( glBindVertexArray,          PFNGLBINDVERTEXARRAYPROC         );
    GLDECLAREFUNCTION ( glDeleteVertexArrays,       PFNGLDELETEVERTEXARRAYSPROC      );

    static char log_buffer[1024] = {0};

    static void LoadOpenGLFunctions()
    {
        GLGETPROCADDRESS ( glAttachShader,             PFNGLATTACHSHADERPROC            );
        GLGETPROCADDRESS ( glCompileShader,            PFNGLCOMPILESHADERPROC           );
        GLGETPROCADDRESS ( glCreateProgram,            PFNGLCREATEPROGRAMPROC           );
        GLGETPROCADDRESS ( glCreateShader,             PFNGLCREATESHADERPROC            );
        GLGETPROCADDRESS ( glDeleteProgram,            PFNGLDELETEPROGRAMPROC           );
        GLGETPROCADDRESS ( glDeleteShader,             PFNGLDELETESHADERPROC            );
        GLGETPROCADDRESS ( glEnableVertexAttribArray,  PFNGLENABLEVERTEXATTRIBARRAYPROC );
        GLGETPROCADDRESS ( glGetAttribLocation,        PFNGLGETATTRIBLOCATIONPROC       );
        GLGETPROCADDRESS ( glGetProgramiv,             PFNGLGETPROGRAMIVPROC            );
        GLGETPROCADDRESS ( glGetProgramInfoLog,        PFNGLGETPROGRAMINFOLOGPROC       );
        GLGETPROCADDRESS ( glGetShaderiv,              PFNGLGETSHADERIVPROC             );
        GLGETPROCADDRESS ( glGetShaderInfoLog,         PFNGLGETSHADERINFOLOGPROC        );
        GLGETPROCADDRESS ( glGetUniformLocation,       PFNGLGETUNIFORMLOCATIONPROC      );
        GLGETPROCADDRESS ( glLinkProgram,              PFNGLLINKPROGRAMPROC             );
        GLGETPROCADDRESS ( glShaderSource,             PFNGLSHADERSOURCEPROC            );
        GLGETPROCADDRESS ( glUseProgram,               PFNGLUSEPROGRAMPROC              );
        GLGETPROCADDRESS ( glUniform1i,                PFNGLUNIFORM1IPROC               );
        GLGETPROCADDRESS ( glUniformMatrix4fv,         PFNGLUNIFORMMATRIX4FVPROC        );
        GLGETPROCADDRESS ( glVertexAttribPointer,      PFNGLVERTEXATTRIBPOINTERPROC     );
        GLGETPROCADDRESS ( glGenBuffers,               PFNGLGENBUFFERSPROC              );
        GLGETPROCADDRESS ( glBindBuffer,               PFNGLBINDBUFFERPROC              );
        GLGETPROCADDRESS ( glBufferData,               PFNGLBUFFERDATAPROC              );
        GLGETPROCADDRESS ( glDeleteBuffers,            PFNGLDELETEBUFFERSPROC           );
        GLGETPROCADDRESS ( glGenVertexArrays,          PFNGLGENVERTEXARRAYSPROC         );
        GLGETPROCADDRESS ( glBindVertexArray,          PFNGLBINDVERTEXARRAYPROC         );
        GLGETPROCADDRESS ( glDeleteVertexArrays,       PFNGLDELETEVERTEXARRAYSPROC      );
    }

    OpenGLRenderer::OpenGLRenderer() :
        screen_texture ( 0 ),
        max_texture_size ( 0 ),
        vert_shader ( 0 ), frag_shader ( 0 ),
        shader_program ( 0 ),
        vertex_buffer_object ( 0 ),
        vertex_array_object ( 0 )
    {
    }

    bool OpenGLRenderer::ChangeScreenSize ( int32_t screen_width, int32_t screen_height )
    {
        Renderer::ChangeScreenSize ( screen_width, screen_height );
        glUseProgram ( shader_program );
        LOGERROR();
        GLfloat projection[16];
        float width;
        float height;

        const float pixel_offset = -0.375f;

        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
        }

        glGetIntegerv ( GL_MAX_TEXTURE_SIZE, &max_texture_size );
        LOGERROR();
        if ( ( screen_width > max_texture_size ) ||
             ( screen_height > max_texture_size ) )
        {
#ifdef WIN32
            printf ( "Error: %s\n", "Screen texture dimensions surpass maximum allowed OpenGL texture size" );
#else
            printf ( "\033[31mError:\033[0m %s\n", "Screen texture dimensions surpass maximum allowed OpenGL texture size" );
#endif
            return false;
        }
        glGenTextures ( 1, &screen_texture );
        LOGERROR();
        glBindTexture ( GL_TEXTURE_2D, screen_texture );
        LOGERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        LOGERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        LOGERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        LOGERROR();
        glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        LOGERROR();
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL );
        LOGERROR();

        width =  float ( screen_w ) + pixel_offset;
        height = float ( screen_h ) + pixel_offset;

        // glOrtho
        // left          right  bottom  top           near  far
        // pixel_offset, width, height, pixel_offset, 0.0f, 1.0f
        // X
        projection[0 ] =  2.0f / ( width - pixel_offset );
        projection[1 ] =  0.0f;
        projection[2 ] =  0.0f;
        projection[3 ] =  0.0f;

        // Y
        projection[4 ] =  0.0f;
        projection[5 ] =  2.0f / ( pixel_offset - height );
        projection[6 ] =  0.0f;
        projection[7 ] =  0.0f;

        // Z
        projection[8 ]  =  0.0f;
        projection[9 ]  =  0.0f;
        projection[10]  = -2.0f;
        projection[11]  =  0.0f;

        // Pos
        projection[12] = - ( ( width + pixel_offset ) / ( width - pixel_offset ) );
        projection[13] = - ( ( pixel_offset + height ) / ( pixel_offset - height ) );
        projection[14] = -1.0f;
        projection[15] =  1.0f;

        GLint projection_matrix = glGetUniformLocation ( shader_program, "projection_matrix" );
        LOGERROR();
        if ( projection_matrix > -1 )
        {
            glUniformMatrix4fv ( projection_matrix, 1, GL_FALSE, projection );
            LOGERROR();
        }

        GLint screen_texture_index = glGetUniformLocation ( shader_program, "screen_texture" );
        LOGERROR();
        if ( screen_texture_index > -1 )
        {
            glUniform1i ( screen_texture_index, 0 );
            LOGERROR();
        }

        // Create VBO
        GLfloat vertices[16] =
        {
            /* position */ 0.0f, 0.0f,                                                                /* uv */ 0.0f, 0.0f,
            /* position */ static_cast<float> ( screen_width ), 0.0f,                                 /* uv */ 1.0f, 0.0f,
            /* position */ 0.0f, static_cast<float> ( screen_height ),                                /* uv */ 0.0f, 1.0f,
            /* position */ static_cast<float> ( screen_width ), static_cast<float> ( screen_height ), /* uv */ 1.0f, 1.0f
        };


        glBindVertexArray ( vertex_array_object );
        LOGERROR();

        glBindBuffer ( GL_ARRAY_BUFFER, vertex_buffer_object );
        LOGERROR();

        glBufferData ( GL_ARRAY_BUFFER, sizeof ( GLfloat ) * 16, &vertices[0], GL_STATIC_DRAW );
        LOGERROR();

        return true;
    }

    bool OpenGLRenderer::Initialize ()
    {
        LoadOpenGLFunctions();

        // Compile Shaders
        GLint info_log_length;
        GLint shader_code_length;
        LOGERROR ();
        if ( ( vert_shader = glCreateShader ( GL_VERTEX_SHADER ) ) == 0 )
        {
            LOGERROR ();
            return false;
        }
        if ( ( frag_shader = glCreateShader ( GL_FRAGMENT_SHADER ) ) == 0 )
        {
            LOGERROR ();
            return false;
        }

        shader_code_length = static_cast<GLint> ( strlen ( vertex_shader ) );
        glShaderSource ( vert_shader, 1, ( const char ** ) &vertex_shader, &shader_code_length );
        LOGERROR();
        shader_code_length = static_cast<GLint> ( strlen ( fragment_shader ) );
        glShaderSource ( frag_shader, 1, ( const char ** ) &fragment_shader, &shader_code_length );
        LOGERROR();


        glCompileShader ( vert_shader );
        LOGERROR();

        glCompileShader ( frag_shader );
        LOGERROR();

        GLint compile_status;

        glGetShaderiv ( vert_shader, GL_COMPILE_STATUS, &compile_status );
        LOGERROR();
        glGetShaderiv ( vert_shader, GL_INFO_LOG_LENGTH, &info_log_length );
        LOGERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( vert_shader, info_log_length, NULL, log_buffer );
#ifdef WIN32
            printf ( "Error: %s\n", log_buffer );
#else
            printf ( "\033[31mError:\033[0m %s\n", log_buffer );
#endif
        }

        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        glGetShaderiv ( frag_shader, GL_COMPILE_STATUS, &compile_status );
        LOGERROR();
        glGetShaderiv ( frag_shader, GL_INFO_LOG_LENGTH, &info_log_length );
        LOGERROR();

        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetShaderInfoLog ( frag_shader, info_log_length, NULL, log_buffer );
#ifdef WIN32
            printf ( "Error: %s\n", log_buffer );
#else
            printf ( "\033[31mError:\033[0m %s\n", log_buffer );
#endif
        }
        if ( compile_status != GL_TRUE )
        {
            return false;
        }

        // Create Shader Program
        shader_program = glCreateProgram();
        LOGERROR();
        GLint link_status;

        glAttachShader ( shader_program, vert_shader );
        LOGERROR();
        glAttachShader ( shader_program, frag_shader );
        LOGERROR();

        glLinkProgram ( shader_program );
        LOGERROR();

        glGetProgramiv ( shader_program, GL_LINK_STATUS, &link_status );
        LOGERROR();
        glGetProgramiv ( shader_program, GL_INFO_LOG_LENGTH, &info_log_length );
        LOGERROR();
        if ( info_log_length > 1 )
        {
            info_log_length = std::min ( info_log_length, 1024 );
            glGetProgramInfoLog ( shader_program, info_log_length, NULL, log_buffer );
            LOGERROR();
#ifdef WIN32
            printf ( "Error: %s\n", log_buffer );
#else
            printf ( "\033[31mError:\033[0m %s\n", log_buffer );
#endif
        }
        if ( link_status != GL_TRUE )
        {
            return false;
        }
        return true;

        // Generate VAO
        glGenVertexArrays ( 1, &vertex_array_object );
        LOGERROR();

        // Generate VBO
        glGenBuffers ( 1, &vertex_buffer_object );
        LOGERROR();

    }

    void OpenGLRenderer::Finalize()
    {
        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
            screen_texture = 0;
        }

        if ( shader_program > 0 )
        {
            glDeleteProgram ( shader_program );
            shader_program = 0;
        }

        if ( vert_shader > 0 )
        {
            glDeleteShader ( vert_shader );
            vert_shader = 0;
        }

        if ( frag_shader > 0 )
        {
            glDeleteShader ( frag_shader );
            frag_shader = 0;
        }

        if ( vertex_array_object != 0 )
        {
            glDeleteVertexArrays ( 1, &vertex_array_object );
            LOGERROR();
            vertex_array_object = 0;
        }
        if ( vertex_buffer_object != 0 )
        {
            glDeleteBuffers ( 1, &vertex_buffer_object );
            LOGERROR();
            vertex_buffer_object = 0;
        }
        Renderer::Finalize();
    }

    void OpenGLRenderer::BeginRender()
    {
        ///\todo Setting the screen bitmap memory to zero may not be always necessary.
        memset ( reinterpret_cast<void*> ( screen_bitmap ), 0x00, sizeof ( uint8_t ) * ( screen_w * screen_h * 4 ) );
    }

    void OpenGLRenderer::EndRender()
    {
        glUseProgram ( shader_program );
        LOGERROR();

        glBindVertexArray ( vertex_array_object );
        LOGERROR();

        glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        LOGERROR();

        glEnable ( GL_BLEND );
        LOGERROR();

        glDepthMask ( GL_FALSE );
        glDisable ( GL_DEPTH_TEST );

        glBindTexture ( GL_TEXTURE_2D, screen_texture );
        LOGERROR();

        glTexSubImage2D ( GL_TEXTURE_2D, 0, 0, 0, screen_w, screen_h, GL_BGRA, GL_UNSIGNED_BYTE, screen_bitmap );
        LOGERROR();

        GLint position = glGetAttribLocation ( shader_program, "position" );
        LOGERROR();
        glVertexAttribPointer ( position, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, ( ( void* ) 0 ) );
        LOGERROR();
        glEnableVertexAttribArray ( position );
        LOGERROR();

        GLint uv = glGetAttribLocation ( shader_program, "uv" );
        glVertexAttribPointer ( uv, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, ( ( void* ) ( sizeof ( float ) * 2 ) ) );
        LOGERROR();
        glEnableVertexAttribArray ( uv );
        LOGERROR();

        glDrawArrays ( GL_TRIANGLE_STRIP, 0, 4 );
        LOGERROR();
    }
}
