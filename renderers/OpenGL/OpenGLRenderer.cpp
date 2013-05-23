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
#endif
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

    static PFNGLATTACHSHADERPROC            glAttachShader = NULL;
    static PFNGLCOMPILESHADERPROC           glCompileShader = NULL;
    static PFNGLCREATEPROGRAMPROC           glCreateProgram = NULL;
    static PFNGLCREATESHADERPROC            glCreateShader = NULL;
    static PFNGLDELETEPROGRAMPROC           glDeleteProgram = NULL;
    static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = NULL;
    static PFNGLGETATTRIBLOCATIONPROC       glGetAttribLocation = NULL;
    static PFNGLGETPROGRAMIVPROC            glGetProgramiv = NULL;
    static PFNGLGETPROGRAMINFOLOGPROC       glGetProgramInfoLog = NULL;
    static PFNGLGETSHADERIVPROC             glGetShaderiv = NULL;
    static PFNGLGETSHADERINFOLOGPROC        glGetShaderInfoLog = NULL;
    static PFNGLGETUNIFORMLOCATIONPROC      glGetUniformLocation = NULL;
    static PFNGLLINKPROGRAMPROC             glLinkProgram = NULL;
    static PFNGLSHADERSOURCEPROC            glShaderSource = NULL;
    static PFNGLUSEPROGRAMPROC              glUseProgram = NULL;
    static PFNGLUNIFORM1IPROC               glUniform1i = NULL;
    static PFNGLUNIFORMMATRIX4FVPROC        glUniformMatrix4fv = NULL;
    static PFNGLVERTEXATTRIBPOINTERPROC     glVertexAttribPointer = NULL;
    static PFNGLGENBUFFERSPROC              glGenBuffers = NULL;
    static PFNGLBINDBUFFERPROC              glBindBuffer = NULL;
    static PFNGLBUFFERDATAPROC              glBufferData = NULL;

    static char log_buffer[1024] = {0};

    uint32_t OpenGLRenderer::TypeTable[] =
    {
        GL_UNSIGNED_BYTE
    };

    uint32_t OpenGLRenderer::FormatTable[] =
    {
        GL_RGB,
        GL_BGR,
        GL_RGBA,
        GL_BGRA,
    };

    OpenGLRenderer::OpenGLRenderer() :
        viewport_w ( 0 ), viewport_h ( 0 ),
        screen_texture ( 0 ), screen_bitmap ( NULL ),
        shader_program ( 0 )
    {
    }

    bool OpenGLRenderer::ChangeScreenSize ( int32_t screen_width, int32_t screen_height )
    {
        glUseProgram ( shader_program );
        LOGERROR();
        GLint viewport[4];
        GLfloat projection[16];
        float width;
        float height;

        const float pixel_offset = -0.375f;

        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
        }
        if ( screen_bitmap != NULL )
        {
            delete[] screen_bitmap;
        }

        glGetIntegerv ( GL_VIEWPORT, viewport );
        LOGERROR();
        viewport_w = viewport[2] - viewport[0];
        viewport_h = viewport[3] - viewport[1];

        screen_w = screen_width;
        screen_h = screen_height;

        glGetIntegerv ( GL_MAX_TEXTURE_SIZE, &max_texture_size );
        LOGERROR();
        if ( ( screen_w > max_texture_size ) ||
             ( screen_h > max_texture_size ) )
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
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8, screen_w, screen_h, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL );
        LOGERROR();
        screen_bitmap = new uint8_t[screen_w * screen_h * 4];

        width =  float ( viewport_w ) + pixel_offset;
        height = float ( viewport_h ) + pixel_offset;

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

        GLint screen_texture = glGetUniformLocation ( shader_program, "screen_texture" );
        LOGERROR();
        if ( screen_texture > -1 )
        {
            glUniform1i ( screen_texture, 1 );
            LOGERROR();
        }

        // Create VBO
        GLfloat vertices[16] =
        {
            /* position */ 0.0f, 0.0f,                                                 /* uv */ 0.0f, 0.0f,
            /* position */ static_cast<float> ( screen_w ), 0.0f,                         /* uv */ 1.0f, 0.0f,
            /* position */ 0.0f, static_cast<float> ( screen_h ),                         /* uv */ 0.0f, 1.0f,
            /* position */ static_cast<float> ( screen_w ), static_cast<float> ( screen_h ), /* uv */ 1.0f, 1.0f
        };

        // Generate VBO
        glGenBuffers ( 1, &vertex_buffer_object );
        LOGERROR();
        glBindBuffer ( GL_ARRAY_BUFFER, vertex_buffer_object );
        LOGERROR();
        glBufferData ( GL_ARRAY_BUFFER, sizeof ( GLfloat ) * 16, vertices, GL_STATIC_DRAW );
        LOGERROR();

        return true;
    }

    bool OpenGLRenderer::Initialize ( int32_t screen_width, int32_t screen_height )
    {
        glAttachShader =            ( PFNGLATTACHSHADERPROC )            wglGetProcAddress ( "glAttachShader" );
        glCompileShader =           ( PFNGLCOMPILESHADERPROC )            wglGetProcAddress ( "glCompileShader" );
        glCreateProgram =           ( PFNGLCREATEPROGRAMPROC )           wglGetProcAddress ( "glCreateProgram" );
        glCreateShader =            ( PFNGLCREATESHADERPROC  )           wglGetProcAddress ( "glCreateShader" );
        glDeleteProgram =           ( PFNGLDELETEPROGRAMPROC )           wglGetProcAddress ( "glDeleteProgram" );
        glEnableVertexAttribArray = ( PFNGLENABLEVERTEXATTRIBARRAYPROC ) wglGetProcAddress ( "glEnableVertexAttribArray" );
        glGetAttribLocation =       ( PFNGLGETATTRIBLOCATIONPROC )       wglGetProcAddress ( "glGetAttribLocation" );
        glGetProgramiv =            ( PFNGLGETPROGRAMIVPROC )            wglGetProcAddress ( "glGetProgramiv" );
        glGetProgramInfoLog =       ( PFNGLGETPROGRAMINFOLOGPROC )       wglGetProcAddress ( "glGetProgramInfoLog" );
        glGetShaderiv =             ( PFNGLGETSHADERIVPROC )             wglGetProcAddress ( "glGetShaderiv" );
        glGetShaderInfoLog =        ( PFNGLGETSHADERINFOLOGPROC )        wglGetProcAddress ( "glGetShaderInfoLog" );
        glGetUniformLocation =      ( PFNGLGETUNIFORMLOCATIONPROC )      wglGetProcAddress ( "glGetUniformLocation" );
        glLinkProgram =             ( PFNGLLINKPROGRAMPROC )             wglGetProcAddress ( "glLinkProgram" );
        glShaderSource =            ( PFNGLSHADERSOURCEPROC )            wglGetProcAddress ( "glShaderSource" );
        glUseProgram =              ( PFNGLUSEPROGRAMPROC )              wglGetProcAddress ( "glUseProgram" );
        glUniform1i =               ( PFNGLUNIFORM1IPROC )               wglGetProcAddress ( "glUniform1i" );
        glUniformMatrix4fv =        ( PFNGLUNIFORMMATRIX4FVPROC )        wglGetProcAddress ( "glUniformMatrix4fv" );
        glVertexAttribPointer =     ( PFNGLVERTEXATTRIBPOINTERPROC )     wglGetProcAddress ( "glVertexAttribPointer" );
        glGenBuffers =              ( PFNGLGENBUFFERSPROC )              wglGetProcAddress ( "glGenBuffers" );
        glBindBuffer =              ( PFNGLBINDBUFFERPROC )              wglGetProcAddress ( "glBindBuffer" );
        glBufferData =              ( PFNGLBUFFERDATAPROC )              wglGetProcAddress ( "glBufferData" );

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

        if ( !ChangeScreenSize ( screen_width, screen_height ) )
        {
            return false;
        }
        LOGERROR();
        return true;
    }

    void OpenGLRenderer::Finalize()
    {
        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
            screen_texture = 0;
        }
        if ( screen_bitmap != NULL )
        {
            delete[] screen_bitmap;
            screen_bitmap = NULL;
        }
        if ( shader_program != 0 )
        {
            glDeleteProgram ( shader_program );
            LOGERROR();
            shader_program = 0;
        }
    }

    void OpenGLRenderer::BeginRender()
    {
        glUseProgram ( shader_program );
        LOGERROR();
        ///\todo Setting the screen bitmap memory to zero may not be always necesary.
        memset ( screen_bitmap, 0, sizeof ( uint8_t ) * ( screen_w * screen_h * 4 ) );
    }

    void OpenGLRenderer::EndRender()
    {
        glBindBuffer ( GL_ARRAY_BUFFER, vertex_buffer_object );
        LOGERROR();

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

        //glPopAttrib();
        //LOGERROR();
    }
    void OpenGLRenderer::DrawRect ( Color color, const Rect* rect )
    {
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );
        int32_t x1 = ( rect->GetLeft() < 0 ) ? 0 : rect->GetLeft();
        int32_t x2 = ( rect->GetRight() > screen_w ) ? screen_w : rect->GetRight();
        int32_t y1 = ( rect->GetTop() < 0 ) ? 0 : rect->GetTop();
        int32_t y2 = ( rect->GetBottom() > screen_h ) ? screen_h : rect->GetBottom();

        if ( ( x1 > screen_w ) || ( x2 < 0 ) ||
             ( y1 > screen_h ) || ( y2 < 0 ) )
        {
            return;
        }
        for ( int32_t y = y1; y < y2; ++y )
        {
            for ( int32_t x = x1; x < x2; ++x )
            {
                pixels[ ( y * screen_w ) + x].Blend ( color );
            }
        }
    }
    void OpenGLRenderer::DrawRectOutline ( Color color, const Rect* rect )
    {
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );
        int32_t x1 = ( rect->GetLeft() < 0 ) ? 0 : rect->GetLeft();
        int32_t x2 = ( rect->GetRight() > screen_w ) ? ( screen_w - 1 ) : ( rect->GetRight() - 1 );
        int32_t y1 = ( rect->GetTop() < 0 ) ? 0 : rect->GetTop();
        int32_t y2 = ( rect->GetBottom() > screen_h ) ? ( screen_h - 1 ) : ( rect->GetBottom() - 1 );

        if ( ( x1 > screen_w ) || ( x2 < 0 ) ||
             ( y1 > screen_h ) || ( y2 < 0 ) )
        {
            return;
        }

        for ( int32_t y = y1; y <= y2; ++y )
        {
            pixels[ ( y * screen_w ) + x1].Blend ( color );
            pixels[ ( y * screen_w ) + x2].Blend ( color );
        }
        // Avoid setting the corner pixels twice
        for ( int32_t x = x1 + 1; x < x2; ++x )
        {
            pixels[ ( y1 * screen_w ) + x].Blend ( color );
            pixels[ ( y2 * screen_w ) + x].Blend ( color );
        }
    }

    void OpenGLRenderer::DrawImage ( Color color, int32_t x, int32_t y, Image* image )
    {
        assert ( image != NULL );
        const Color* image_bitmap = image->GetBitmap();
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );

        int32_t x1 = x;
        int32_t x2 = x + image->GetWidth();
        int32_t y1 = y;
        int32_t y2 = y + image->GetHeight();

        if ( ( x1 > screen_w ) || ( x2 < 0 ) ||
             ( y1 > screen_h ) || ( y2 < 0 ) )
        {
            return;
        }

        int32_t iy = 0;
        for ( int32_t sy = y1; sy < y2; ++sy )
        {
            if ( ( sy >= 0 ) && ( sy < screen_h ) )
            {
                int32_t ix = 0;
                for ( int32_t sx = x1; sx < x2; ++sx )
                {
                    if ( ( sx >= 0 ) && ( sx < screen_w ) )
                    {
                        pixels[ ( ( sy * screen_w ) + sx )].Blend ( image_bitmap[ ( ( iy * image->GetWidth() ) + ix )] );
                    }
                    ++ix;
                }
            }
            ++iy;
        }
    }

    void OpenGLRenderer::DrawString ( Color color, int32_t x, int32_t y, const wchar_t* text )
    {
        size_t textlength = wcslen ( text );
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );
        int32_t curx;
        int32_t cury;
        Font::Glyph* glyph;
        const uint8_t* glyph_map = font->GetGlyphMap();
        int32_t glyph_map_width = font->GetMapWidth();
        Color pixel;
        pixel.r = color.r;
        pixel.g = color.g;
        pixel.b = color.b;
        for ( size_t i = 0; i < textlength; ++i )
        {
            // Find Character Code
            glyph = font->GetGlyph ( text[i] );

            if ( glyph == NULL )
            {
                continue;
            };

            cury = y - glyph->top;
            for ( int32_t Y = glyph->min[1]; Y < glyph->max[1]; ++Y )
            {
                if ( ( cury >= 0 ) && ( cury < screen_h ) )
                {
                    curx = x + glyph->left;
                    for ( int32_t X = glyph->min[0]; X < ( glyph->max[0] ); ++X )
                    {
                        if ( ( curx >= 0 ) && ( curx < screen_w ) )
                        {
                            pixel.a = glyph_map[ ( Y * glyph_map_width ) + X];
                            pixels[ ( cury * screen_w ) + curx ].Blend ( pixel );
                        }
                        ++curx;
                    }
                }
                ++cury;
            }
            // Advance pen position
            x += glyph->advance[0];
        }
    }
}
