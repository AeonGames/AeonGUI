/******************************************************************************
Copyright 2010-2013 Rodrigo Hernandez Cordoba

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
#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include "Renderer.h"
#include "Widget.h"

namespace AeonGUI
{

    Renderer::Renderer() : font ( NULL ), screen_w ( 0 ), screen_h ( 0 ), screen_bitmap ( NULL ), widgets ( NULL )
    {
    }

    Renderer::~Renderer()
    {
    }

    bool Renderer::Initialize ( )
    {
        return true;
    }

    void Renderer::Finalize()
    {
        font = NULL;
        screen_w = 0;
        screen_h = 0;
        if ( screen_bitmap != NULL )
        {
            delete[] screen_bitmap;
            screen_bitmap = NULL;
        }
    }

    bool Renderer::ChangeScreenSize ( int32_t screen_width, int32_t screen_height )
    {
        screen_w = screen_width;
        screen_h = screen_height;
        if ( screen_bitmap != NULL )
        {
            delete[] screen_bitmap;
        }
        screen_bitmap = new uint8_t[screen_w * screen_h * 4];
        return true;
    }

    void Renderer::SetFont ( Font* newfont )
    {
        font = newfont;
    }

    const Font* Renderer::GetFont()
    {
        return font;
    }

    void Renderer::DrawRect ( Color color, const Rect* rect )
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
    void Renderer::DrawRectOutline ( Color color, const Rect* rect )
    {
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );
        int32_t x1 = ( rect->GetLeft() < 0 ) ? 0 : rect->GetLeft();
        int32_t x2 = ( rect->GetRight() > screen_w ) ? ( screen_w - 1 ) : ( rect->GetRight() - 1 );
        int32_t y1 = ( rect->GetTop() < 0 ) ? 0 : rect->GetTop();
        int32_t y2 = ( rect->GetBottom() > screen_h ) ? ( screen_h - 1 ) : ( rect->GetBottom() - 1 );

        if ( ( x1 >= screen_w ) || ( x2 < 0 ) ||
             ( y1 >= screen_h ) || ( y2 < 0 ) )
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

    static float LanczosKernel ( float  x )
    {
        const float filter = 3.0f;
        const float fx = fabsf ( x );
        if ( fx == 0.0f )
        {
            return ( 1.0f );
        }
        else if ( fx >= filter )
        {
            return ( 0.0f );
        }
        const float pix = fx * static_cast<float> ( M_PI );
        const float pix_over_filter = pix / filter;
        return ( sinf ( pix ) / pix ) * ( sinf ( pix_over_filter ) / ( pix_over_filter ) );
    }

    static Color Lanczos1DInterpolation ( float x, const Color* samples, int32_t sample_count, uint32_t sample_stride )
    {
        const int32_t filter = 3;
        int32_t fx = static_cast<int32_t> ( floorf ( x ) );
        Color result = 0;
        float sum = 0;
        float b = 0, g = 0, r = 0, a = 0;
        float kernel[ ( filter * 2 ) - 1];

        for ( int32_t i = -2; i < 3; ++i )
        {
            sum += kernel[i + 2] = LanczosKernel ( x - ( fx + i ) );
        }

        // Normalize
        for ( int32_t i = 0; i < ( filter * 2 ) - 1; ++i )
        {
            kernel[i] /= sum;
        }

        for ( int32_t i = -2; i < 3; ++i )
        {
            if ( ( fx + i ) < 0 )
            {
                b += samples[0].b * kernel[i + 2];
                g += samples[0].g * kernel[i + 2];
                r += samples[0].r * kernel[i + 2];
                a += samples[0].a * kernel[i + 2];
            }
            else if ( ( fx + i ) >= sample_count )
            {
                b += samples[ ( sample_count - 1 ) * sample_stride].b * kernel[i + 2];
                g += samples[ ( sample_count - 1 ) * sample_stride].g * kernel[i + 2];
                r += samples[ ( sample_count - 1 ) * sample_stride].r * kernel[i + 2];
                a += samples[ ( sample_count - 1 ) * sample_stride].a * kernel[i + 2];
            }
            else
            {
                b += samples[ ( fx + i ) * sample_stride].b * kernel[i + 2];
                g += samples[ ( fx + i ) * sample_stride].g * kernel[i + 2];
                r += samples[ ( fx + i ) * sample_stride].r * kernel[i + 2];
                a += samples[ ( fx + i ) * sample_stride].a * kernel[i + 2];
            }
        }
        result.b = ( b < 0.0f ) ? 0 : ( b > 255.0f ) ? 255 : static_cast<uint8_t> ( b );
        result.g = ( g < 0.0f ) ? 0 : ( g > 255.0f ) ? 255 : static_cast<uint8_t> ( g );
        result.r = ( r < 0.0f ) ? 0 : ( r > 255.0f ) ? 255 : static_cast<uint8_t> ( r );
        result.a = ( a < 0.0f ) ? 0 : ( a > 255.0f ) ? 255 : static_cast<uint8_t> ( a );
        return result;
    }

    static Color Lanczos2DInterpolation ( float x, float y, uint32_t w, uint32_t h, const Color* buffer )
    {
        const int32_t filter = 3;
        int32_t fx = static_cast<int32_t> ( floorf ( x ) );
        int32_t fy = static_cast<int32_t> ( floorf ( y ) );
        Color result = 0;
        float sumx = 0;
        float sumy = 0;

        float finalsums[4] = {0};
        float kernelx[ ( filter * 2 ) - 1];
        float kernely[ ( filter * 2 ) - 1];

        for ( int32_t i = -2; i < 3; ++i )
        {
            sumx += kernelx[i + 2] = LanczosKernel ( x - ( fx + i ) );
            sumy += kernely[i + 2] = LanczosKernel ( y - ( fy + i ) );
        }

        // Normalize
        for ( int32_t i = 0; i < ( filter * 2 ) - 1; ++i )
        {
            kernelx[i] /= sumx;
            kernely[i] /= sumy;
        }

        for ( int32_t yi = -2; yi < 3; ++yi )
        {
            int32_t row = fy + yi;
            if ( row < 0 )
            {
                row = 0;
            }
            else if ( row >= static_cast<int32_t> ( h ) )
            {
                row = h - 1;
            }
            float sums[4] = {0};

            for ( int32_t xi = -2; xi < 3; ++xi )
            {
                int32_t column = fx + xi;
                if ( column < 0 )
                {
                    column = 0;
                }
                else if ( column >= static_cast<int32_t> ( w ) )
                {
                    column = w - 1;
                }
                sums[0] += buffer[ ( row * w ) + column].b * kernelx[xi + 2];
                sums[1] += buffer[ ( row * w ) + column].g * kernelx[xi + 2];
                sums[2] += buffer[ ( row * w ) + column].r * kernelx[xi + 2];
                sums[3] += buffer[ ( row * w ) + column].a * kernelx[xi + 2];
            }
            sums[0] *= kernely[yi + 2];
            sums[1] *= kernely[yi + 2];
            sums[2] *= kernely[yi + 2];
            sums[3] *= kernely[yi + 2];

            finalsums[0] += sums[0];
            finalsums[1] += sums[1];
            finalsums[2] += sums[2];
            finalsums[3] += sums[3];
        }

        result.b = ( finalsums[0] < 0.0f ) ? 0 : ( finalsums[0] > 255.0f ) ? 255 : static_cast<uint8_t> ( finalsums[0] );
        result.g = ( finalsums[1] < 0.0f ) ? 0 : ( finalsums[1] > 255.0f ) ? 255 : static_cast<uint8_t> ( finalsums[1] );
        result.r = ( finalsums[2] < 0.0f ) ? 0 : ( finalsums[2] > 255.0f ) ? 255 : static_cast<uint8_t> ( finalsums[2] );
        result.a = ( finalsums[3] < 0.0f ) ? 0 : ( finalsums[3] > 255.0f ) ? 255 : static_cast<uint8_t> ( finalsums[3] );

        return result;
    }

    static Color NearestNeighbor1DInterpolation ( float x, const Color* samples, int32_t sample_count, uint32_t sample_stride )
    {
        int32_t fx = static_cast<int32_t> ( floorf ( x ) );
        if ( ( x - fx ) > 0.5 )
        {
            ++fx;
        }
        return samples[ fx * sample_stride];
    }

    static Color NearestNeighbor2DInterpolation ( float x, float y, uint32_t w, uint32_t h, const Color* buffer )
    {
        assert ( x < w );
        assert ( y < h );
        int32_t fx = static_cast<int32_t> ( floorf ( x ) );
        int32_t fy = static_cast<int32_t> ( floorf ( y ) );
        float   dx = ( x - fx );
        float   dy = ( y - fy );
        if ( dx > 0.5 )
        {
            ++fx;
        }
        if ( dy > 0.5 )
        {
            ++fy;
        }
        return buffer[ ( fy * w ) + fx];
    }

    void Renderer::DrawImage ( Image* image, int32_t x, int32_t y, int32_t w, int32_t h, ResizeAlgorithm algorithm )
    {
        uint32_t image_w = image->GetWidth();
        uint32_t image_h = image->GetHeight();

        uint32_t stretch_x = image->GetStretchX();
        uint32_t stretch_y = image->GetStretchY();
        uint32_t stretch_width = image->GetStretchWidth();
        uint32_t stretch_height = image->GetStretchHeight();

        if ( ( stretch_x == 0 ) && ( stretch_y == 0 ) && ( stretch_width == image_w ) && ( stretch_height == image_h ) )
        {
            DrawSubImage ( image, x, y, 0, 0, 0, 0, w, h, algorithm );
            return;
        }

        // Draw each 9 patch
        ///\todo Calculate scaled stretch width and height to draw remaining patches.
        DrawSubImage ( image, x, y, 0, 0, stretch_x, stretch_y, 0, 0, algorithm );
    }

    void Renderer::DrawSubImage ( Image* image, int32_t x, int32_t y, int32_t subx, int32_t suby, int32_t subw, int32_t subh, int32_t w, int32_t h, ResizeAlgorithm algorithm )
    {
        assert ( image != NULL );

        Color ( *Function1DInterpolation ) ( float, const Color*, int32_t, uint32_t );
        Color ( *Function2DInterpolation ) ( float, float, uint32_t, uint32_t, const Color* );

        uint32_t image_w = image->GetWidth();
        uint32_t image_h = image->GetHeight();

        switch ( algorithm )
        {
        case NEAREST:
            Function1DInterpolation = NearestNeighbor1DInterpolation;
            Function2DInterpolation = NearestNeighbor2DInterpolation;
            break;
        case LANCZOS:
            Function1DInterpolation = Lanczos1DInterpolation;
            Function2DInterpolation = Lanczos2DInterpolation;
            break;
        }

        if ( subw == 0 )
        {
            subw = image_w - subx;
        }
        if ( subh == 0 )
        {
            subh = image_h - suby;
        }
        if ( w == 0 )
        {
            w = subw;
        }
        if ( h == 0 )
        {
            h = subh;
        }
        const Color* image_bitmap = image->GetBitmap();
        Color* pixels = reinterpret_cast<Color*> ( screen_bitmap );

        int32_t x1 = x;
        int32_t x2 = x + w;
        int32_t y1 = y;
        int32_t y2 = y + h;

        if ( ( x1 > screen_w ) || ( x2 < 0 ) ||
             ( y1 > screen_h ) || ( y2 < 0 ) )
        {
            return;
        }

        if ( ( w == subw ) && ( h == subh ) )
        {
            // Same scale as original
            for ( int32_t sy = y1, iy = suby; sy < y2; ++sy, ++iy )
            {
                if ( ( sy >= 0 ) && ( sy < screen_h ) )
                {
                    for ( int32_t sx = x1, ix = subx; sx < x2; ++sx, ++ix )
                    {
                        if ( ( sx >= 0 ) && ( sx < screen_w ) )
                        {
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( image_bitmap[ ( ( iy * image_w ) + ix )] );
                        }
                    }
                }
            }
        }
        else if ( w == subw )
        {
            // Verticaly Scaled
            float ratio_h = static_cast<float> ( subh ) / static_cast<float> ( h );
            for ( int32_t sy = y1, stepy = 0; sy < y2; ++sy, ++stepy )
            {
                if ( ( sy >= 0 ) && ( sy < screen_h ) )
                {
                    for ( int32_t sx = x1, ix = subx; sx < x2; ++sx, ++ix )
                    {
                        if ( ( sx >= 0 ) && ( sx < screen_w ) )
                        {
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( Function1DInterpolation ( suby + ( stepy * ratio_h ), image_bitmap + ( ix ), image_h, image_w ) );
                        }
                    }
                }
            }
        }
        else if ( h == subh )
        {
            // Horizontaly Scaled
            float ratio_w = static_cast<float> ( subw ) / static_cast<float> ( w );
            for ( int32_t sy = y1, iy = suby; sy < y2; ++sy, ++iy )
            {
                if ( ( sy >= 0 ) && ( sy < screen_h ) )
                {
                    for ( int32_t sx = x1, stepx = 0; sx < x2; ++sx, ++stepx )
                    {
                        if ( ( sx >= 0 ) && ( sx < screen_w ) )
                        {
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( Function1DInterpolation ( subx + ( stepx * ratio_w ), image_bitmap + ( iy * image_w ), image_w, 1 ) );
                        }
                    }
                }
            }
        }
        else
        {
            // Both Horizontaly and Vertically Scaled
            float ratio_w = static_cast<float> ( subw ) / static_cast<float> ( w );
            float ratio_h = static_cast<float> ( subh ) / static_cast<float> ( h );
            /*
            sx,sy are Screen coordinates.
            ix,iy are Scaled Image coordinates.
            */
            for ( int32_t sy = y1, stepy = 0; sy < y2; ++sy, ++stepy )
            {
                if ( ( sy >= 0 ) && ( sy < screen_h ) )
                {
                    for ( int32_t sx = x1, stepx = 0; sx < x2; ++sx, ++stepx )
                    {
                        if ( ( sx >= 0 ) && ( sx < screen_w ) )
                        {
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( Function2DInterpolation ( subx + ( stepx * ratio_w ), suby + ( stepy * ratio_h ), image_w, image_h, image_bitmap ) );
                        }
                    }
                }
            }
        }
    }

    void Renderer::DrawString ( Color color, int32_t x, int32_t y, const wchar_t* text )
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

    void Renderer::AddWidget ( Widget* widget )
    {
        if ( widget != NULL )
        {
            if ( widgets == NULL )
            {
                widgets = widget;
            }
            else
            {
                Widget* sibling = widgets;
                while ( sibling != NULL )
                {
                    if ( sibling->next == NULL )
                    {
                        sibling->next = widget;
                        sibling = NULL;
                    }
                    else
                    {
                        sibling = sibling->next;
                    }
                }
            }
        }
    }

    void Renderer::RemoveWidget ( Widget* widget )
    {
        assert ( widget != NULL );
        if ( widget == NULL )
        {
            return;
        }
        else if ( widgets->next == NULL )
        {
            widgets = NULL;
        }
        else
        {
            Widget* sibling = widgets;
            while ( sibling != NULL )
            {
                if ( sibling->next == widget )
                {
                    sibling->next = sibling->next->next;
                    widget->next = NULL;
                    sibling = NULL;
                }
                else
                {
                    sibling = sibling->next;
                }
            }
        }
    }

    void Renderer::RenderWidgets()
    {
        Widget* sibling = widgets;
        while ( sibling != NULL )
        {
            sibling->Render ( this );
            sibling = sibling->next;
        }
    }
}
