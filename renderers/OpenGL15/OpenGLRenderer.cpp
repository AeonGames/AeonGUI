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
#endif
#include <cassert>
#include <cstdlib>
#include <cwchar>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "GLee/GLee.h"
#include <GL/glu.h>
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
                            std::cout << "Error " << gluErrorString(error) <<\
                            " at file " << __FILE__ << " at line " <<\
                            __LINE__ << std::endl; \
                        }\
                    }

namespace AeonGUI
{
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
        screen_texture_ratio_w ( 1.0f ),
        screen_texture_ratio_h ( 1.0f )
    {
    }

    void OpenGLRenderer::UpdateScreenSize()
    {

    }

    bool OpenGLRenderer::Initialize ( int32_t screen_width, int32_t screen_height )
    {
        GLint viewport[4];
        glGetIntegerv ( GL_VIEWPORT, viewport );
        viewport_w = viewport[2] - viewport[0];
        viewport_h = viewport[3] - viewport[1];

        screen_w = screen_width;
        screen_h = screen_height;
        if ( GLEE_ARB_texture_non_power_of_two )
        {
            screen_texture_w = screen_w;
            screen_texture_h = screen_h;
        }
        else
        {
            screen_texture_w = 1 << static_cast<int32_t> ( ceil ( ( log ( static_cast<float> ( screen_w ) ) / log ( 2.0f ) ) ) );
            screen_texture_h = 1 << static_cast<int32_t> ( ceil ( ( log ( static_cast<float> ( screen_h ) ) / log ( 2.0f ) ) ) );
        }
        screen_texture_ratio_w = static_cast<float> ( screen_w ) / static_cast<float> ( screen_texture_w );
        screen_texture_ratio_h = static_cast<float> ( screen_h ) / static_cast<float> ( screen_texture_h );
        glGetIntegerv ( GL_MAX_TEXTURE_SIZE, &max_texture_size );
        if ( ( screen_texture_w > max_texture_size ) ||
             ( screen_texture_h > max_texture_size ) )
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
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8, screen_texture_w, screen_texture_h, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL );
        LOGERROR();
        screen_bitmap = new uint8_t[screen_w * screen_h * 4];
        return true;
    }

    void OpenGLRenderer::Finalize()
    {
        if ( screen_texture > 0 )
        {
            glDeleteTextures ( 1, &screen_texture );
        }
        if ( screen_bitmap != NULL )
        {
            delete[] screen_bitmap;
        }
    }

    void OpenGLRenderer::BeginRender()
    {
        float width;
        float height;
        const float pixel_offset = -0.375f;
        glPushAttrib ( GL_ALL_ATTRIB_BITS );

        width =  float ( viewport_w ) + pixel_offset;
        height = float ( viewport_h ) + pixel_offset;

        // GL_PIXEL_MODE_BIT
        glDisable ( GL_TEXTURE_2D );
        glMatrixMode ( GL_PROJECTION );
        glPushMatrix();
        glLoadIdentity();
        glOrtho ( pixel_offset, width, height, pixel_offset, -1.0f, 1.0f );

        glMatrixMode ( GL_MODELVIEW );
        glPushMatrix();
        glLoadIdentity();
        /*
         These should probably be checked and saved somehow, so
         we can truly return to the old state in EndRender.
        */
        glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable ( GL_BLEND );
        glDisable ( GL_DEPTH_TEST );
        memset ( screen_bitmap, 0, sizeof ( uint8_t ) * ( screen_w * screen_h * 4 ) );
    }
    void OpenGLRenderer::EndRender()
    {
        glEnable ( GL_TEXTURE_2D );
        LOGERROR();
        glBindTexture ( GL_TEXTURE_2D, screen_texture );
        LOGERROR();

        glTexSubImage2D ( GL_TEXTURE_2D, 0, 0, 0, screen_w, screen_h, GL_BGRA, GL_UNSIGNED_BYTE, screen_bitmap );
        LOGERROR();

        glColor4ub ( 255, 255, 255, 255 );

        GLint vertices[8] =
        {
            0, 0,
            screen_w, 0,
            0, screen_h,
            screen_w, screen_h
        };
        GLfloat uvs[8] =
        {
            0.0f, 0.0f,
            screen_texture_ratio_w, 0.0f,
            0.0f, screen_texture_ratio_h,
            screen_texture_ratio_w, screen_texture_ratio_h
        };
        glEnableClientState ( GL_VERTEX_ARRAY );
        LOGERROR();
        glEnableClientState ( GL_TEXTURE_COORD_ARRAY );
        LOGERROR();
        glVertexPointer ( 2, GL_INT, 0, vertices );
        LOGERROR();
        glTexCoordPointer ( 2, GL_FLOAT, 0, uvs );
        LOGERROR();
        glDrawArrays ( GL_TRIANGLE_STRIP, 0, 4 );
        LOGERROR();
        glDisableClientState ( GL_TEXTURE_COORD_ARRAY );
        LOGERROR();
        glDisableClientState ( GL_VERTEX_ARRAY );
        LOGERROR();

        glMatrixMode ( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode ( GL_MODELVIEW );
        glPopMatrix();
        glPopAttrib();
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
#if 0
    Image* OpenGLRenderer::NewImage ( uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data )
    {
        return new Image ( width, height, format, type, data );
    }
    void OpenGLRenderer::DeleteImage ( Image* image )
    {
        delete ( Image* ) image;
    }
#endif
}
