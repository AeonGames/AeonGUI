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
#include "Cursor.h"

namespace AeonGUI
{

    Renderer::Renderer() : font ( NULL ), cursor ( NULL ), screen_w ( 0 ), screen_h ( 0 ), screen_bitmap ( NULL ), widgets ( NULL )
    {
    }

    Renderer::~Renderer()
    {
    }

    bool Renderer::Initialize ( )
    {
        return true;
    }

    void Renderer::BeginRender()
    {
        ///\todo Setting the screen bitmap memory to zero may not always be necessary.
        memset ( reinterpret_cast<void*> ( screen_bitmap ), 0x00, sizeof ( uint8_t ) * ( screen_w * screen_h * 4 ) );
    }

    void Renderer::EndRender()
    {
        if ( cursor != NULL )
        {
            cursor->Render ( this );
        }
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

    void Renderer::SetCursor ( Cursor* newcursor )
    {
        cursor = newcursor;
    }

    const Cursor* Renderer::GetCursor()
    {
        return cursor;
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

    void Renderer::DrawImage ( const Image* image, int32_t x, int32_t y, int32_t w, int32_t h, ResizeAlgorithm algorithm )
    {
        assert ( image != NULL );

        uint32_t image_w = image->GetWidth();
        uint32_t image_h = image->GetHeight();

        if ( ( image == NULL ) || ( image_w == 0 ) || ( image_h == 0 ) )
        {
            // Invalid Image
            return;
        }

        uint32_t stretch_x = image->GetStretchXStart();
        uint32_t stretch_y = image->GetStretchYStart();
        uint32_t stretch_width = image->GetStretchWidth();
        uint32_t stretch_height = image->GetStretchHeight();

        if ( ( stretch_x == 0 ) && ( stretch_y == 0 ) && ( stretch_width == 0 ) && ( stretch_height == 0 ) )
        {
            DrawSubImage ( image, x, y, 0, 0, 0, 0, w, h, algorithm );
            return;
        }

        // Draw each 9 patch
        int32_t scaled_stretch_width = w - ( image_w - stretch_width );
        int32_t scaled_stretch_height = h - ( image_h - stretch_height );
        int32_t left_x = x + stretch_x;
        int32_t right_x = left_x + scaled_stretch_width;
        int32_t top_y = y + stretch_y;
        int32_t bottom_y = top_y + scaled_stretch_height;
        int32_t image_right_x = stretch_x + stretch_width;
        int32_t image_right_width = image_w - image_right_x;
        int32_t image_bottom_y = stretch_y + stretch_height;
        int32_t image_bottom_height = image_h - image_bottom_y;

        // Top Left
        DrawSubImage ( image, x, y, 0, 0, stretch_x, stretch_y, 0, 0, algorithm );
        // Top
        DrawSubImage ( image, left_x, y, stretch_x, 0, stretch_width, stretch_y, scaled_stretch_width, 0, algorithm );
        // Top Right
        DrawSubImage ( image, right_x, y, image_right_x, 0, image_right_width, stretch_y, 0, 0, algorithm );
        // Right
        DrawSubImage ( image, right_x, top_y, image_right_x, stretch_y, image_right_width, stretch_height, 0, scaled_stretch_height, algorithm );
        // Bottom Right
        DrawSubImage ( image, right_x, bottom_y, image_right_x, image_bottom_y, image_right_width, image_bottom_height, 0, 0, algorithm );
        // Bottom
        DrawSubImage ( image, left_x, bottom_y, stretch_x, image_bottom_y, stretch_width, image_bottom_height, scaled_stretch_width, 0, algorithm );
        // Bottom Left
        DrawSubImage ( image, x, bottom_y, 0, image_bottom_y, stretch_x, image_bottom_height, 0, 0, algorithm );
        // Left
        DrawSubImage ( image, x, top_y, 0, stretch_y, stretch_x, stretch_height, 0, scaled_stretch_height, algorithm );
        // Center
        DrawSubImage ( image, left_x, top_y, stretch_x, stretch_y, stretch_width, stretch_height, scaled_stretch_width, scaled_stretch_height, algorithm );
    }

#ifndef USE_CUDA
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

    static Color Lanczos1DInterpolation ( int32_t x, int32_t step, float ratio, const Color* samples, int32_t sample_width, uint32_t sample_stride )
    {
        const int32_t filter = 3;
        const int32_t filtertimes2 = filter * 2;
        const int32_t negativefilterminus1 = - ( filter - 1 );
        float fx = ( x + ( ( ( step + 0.5f ) * ratio ) - 0.5f ) );
        int32_t ix = static_cast<int32_t> ( floorf ( fx ) );
        Color result = 0;
        float sum = 0;
        float b = 0, g = 0, r = 0, a = 0;
        float kernel[ filtertimes2 ];

        for ( int32_t i = -2; i <= 3; ++i )
        {
            sum += kernel[i + 2] = LanczosKernel ( fx - ( ix + i ) );
        }

        // Normalize
        for ( int32_t i = 0; i < filtertimes2; ++i )
        {
            kernel[i] /= sum;
        }

        for ( int32_t i = negativefilterminus1; i <= filter; ++i )
        {
            if ( ( ix + i ) < x )
            {
                b += samples[x * sample_stride].b * kernel[i + 2];
                g += samples[x * sample_stride].g * kernel[i + 2];
                r += samples[x * sample_stride].r * kernel[i + 2];
                a += samples[x * sample_stride].a * kernel[i + 2];
            }
            else if ( ( ix + i ) > ( x + sample_width ) - 1 )
            {
                b += samples[ ( ( x + sample_width ) - 1 ) * sample_stride].b * kernel[i + 2];
                g += samples[ ( ( x + sample_width ) - 1 ) * sample_stride].g * kernel[i + 2];
                r += samples[ ( ( x + sample_width ) - 1 ) * sample_stride].r * kernel[i + 2];
                a += samples[ ( ( x + sample_width ) - 1 ) * sample_stride].a * kernel[i + 2];
            }
            else
            {
                b += samples[ ( ix + i ) * sample_stride].b * kernel[i + 2];
                g += samples[ ( ix + i ) * sample_stride].g * kernel[i + 2];
                r += samples[ ( ix + i ) * sample_stride].r * kernel[i + 2];
                a += samples[ ( ix + i ) * sample_stride].a * kernel[i + 2];
            }
        }
        result.b = ( b < 0.0f ) ? 0 : ( b > 255.0f ) ? 255 : static_cast<uint8_t> ( b );
        result.g = ( g < 0.0f ) ? 0 : ( g > 255.0f ) ? 255 : static_cast<uint8_t> ( g );
        result.r = ( r < 0.0f ) ? 0 : ( r > 255.0f ) ? 255 : static_cast<uint8_t> ( r );
        result.a = ( a < 0.0f ) ? 0 : ( a > 255.0f ) ? 255 : static_cast<uint8_t> ( a );
        return result;
    }

    static Color Lanczos2DInterpolation ( int32_t x, int32_t xstep, float xratio, int32_t y, int32_t ystep, float yratio, int32_t w, int32_t h, int32_t pitch, const Color* buffer )
    {
        const int32_t filter = 3;
        float fx = ( x + ( ( ( xstep + 0.5f ) * xratio ) - 0.5f ) );
        float fy = ( y + ( ( ( ystep + 0.5f ) * yratio ) - 0.5f ) );
        int32_t ix = static_cast<int32_t> ( floorf ( fx ) );
        int32_t iy = static_cast<int32_t> ( floorf ( fy ) );
        Color result = 0;
        float sumx = 0;
        float sumy = 0;

        float finalsums[4] = {0};
        float kernelx[ ( filter * 2 ) ];
        float kernely[ ( filter * 2 ) ];

        for ( int32_t i = -2; i <= 3; ++i )
        {
            sumx += kernelx[i + 2] = LanczosKernel ( fx - ( ix + i ) );
            sumy += kernely[i + 2] = LanczosKernel ( fy - ( iy + i ) );
        }

        // Normalize
        for ( int32_t i = 0; i < ( filter * 2 ); ++i )
        {
            kernelx[i] /= sumx;
            kernely[i] /= sumy;
        }

        for ( int32_t yi = -2; yi <= 3; ++yi )
        {
            int32_t row = iy + yi;
            if ( row < y )
            {
                row = y;
            }
            else if ( row >= ( y + h )  )
            {
                row = ( y + h ) - 1;
            }
            float sums[4] = {0};

            for ( int32_t xi = -2; xi <= 3; ++xi )
            {
                int32_t column = ix + xi;
                if ( column < x )
                {
                    column = x;
                }
                else if ( column >= ( x + w )  )
                {
                    column = ( x + w ) - 1;
                }
                sums[0] += buffer[ ( row * pitch ) + column].b * kernelx[xi + 2];
                sums[1] += buffer[ ( row * pitch ) + column].g * kernelx[xi + 2];
                sums[2] += buffer[ ( row * pitch ) + column].r * kernelx[xi + 2];
                sums[3] += buffer[ ( row * pitch ) + column].a * kernelx[xi + 2];
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

    static Color Tile1DInterpolation ( int32_t x, int32_t step, float ratio, const Color* samples, int32_t sample_width, uint32_t sample_stride )
    {
        int32_t ix = x + ( step % sample_width );
        return samples[ ix * sample_stride];
    }

    static Color LinearInterpolation ( int32_t x, int32_t step, float ratio, const Color* samples, int32_t sample_width, uint32_t sample_stride )
    {
        float fx;
        float tx = modff ( x + ( ( ( step + 0.5f ) * ratio ) - 0.5f ), &fx );

        int32_t ix = static_cast<int32_t> ( fx );
        // clamp
        if ( ix >= ( x + sample_width - 1 ) )
        {
            return samples [ ( x + sample_width - 1 ) * sample_stride ];
        }
        else if ( ix < x )
        {
            return samples [ x * sample_stride ];
        }
        const Color* c0 = samples + ( ix * sample_stride );
        const Color* c1 = samples + ( ( ix + 1 ) * sample_stride );
        Color result;
        result.b = static_cast<uint8_t> ( c0->b * ( 1 - tx ) + c1->b * tx );
        result.g = static_cast<uint8_t> ( c0->g * ( 1 - tx ) + c1->g * tx );
        result.r = static_cast<uint8_t> ( c0->r * ( 1 - tx ) + c1->r * tx );
        result.a = static_cast<uint8_t> ( c0->a * ( 1 - tx ) + c1->a * tx );
        return result;
    }

    static Color NearestNeighbor1DInterpolation ( int32_t x, int32_t step, float ratio, const Color* samples, int32_t sample_width, uint32_t sample_stride )
    {
        int32_t ix = static_cast<int32_t> ( floorf ( x + ( ( step + 0.5f ) * ratio ) - 0.5f ) );
        if ( ix >= ( x + sample_width - 1 ) )
        {
            ix = ( x + sample_width - 1 );
        }
        else if ( ix < x )
        {
            ix = x;
        }
        return samples[ ix * sample_stride];
    }

    static Color BilinearInterpolation ( int32_t x, int32_t xstep, float xratio, int32_t y, int32_t ystep, float yratio, int32_t w, int32_t h, int32_t pitch, const Color* buffer )
    {
        float fx;
        float fy;
        float tx = modff ( x + ( ( ( xstep + 0.5f ) * xratio ) - 0.5f ), &fx );
        float ty = modff ( y + ( ( ( ystep + 0.5f ) * yratio ) - 0.5f ), &fy );

        int32_t ix = static_cast<int32_t> ( fx );
        int32_t ix1; // If not clampled should be ix+1
        int32_t iy = static_cast<int32_t> ( fy );
        int32_t iy1; // If not clampled should be iy+1

        // clamp x
        if ( ix >= ( x + w - 1 ) )
        {
            ix1 = ix = ( x + w - 1 );
        }
        else if ( ix < x )
        {
            ix1 = ix = x;
        }
        else
        {
            ix1 = ix + 1;
        }

        // clamp y
        if ( iy >= ( y + h - 1 ) )
        {
            iy1 = iy = ( y + h - 1 );
        }
        else if ( iy < y )
        {
            iy1 = iy = y;
        }
        else
        {
            iy1 = iy + 1;
        }

        const Color* c00 = buffer + ( iy * pitch ) + ix;
        const Color* c10 = buffer + ( iy * pitch ) + ix1;
        const Color* c01 = buffer + ( ( iy1 ) * pitch ) + ix;
        const Color* c11 = buffer + ( ( iy1 ) * pitch ) + ix1;

        Color result;
        // c00 * (1-tx) * (1-ty) + c10 * tx*(1-ty) + c01*(1-tx)*ty+c11*tx*ty
        result.b = static_cast<uint8_t> ( c00->b * ( 1 - tx ) * ( 1 - ty ) + c10->b * tx * ( 1 - ty ) + c01->b * ( 1 - tx ) * ty + c11->b * tx * ty );
        result.g = static_cast<uint8_t> ( c00->g * ( 1 - tx ) * ( 1 - ty ) + c10->g * tx * ( 1 - ty ) + c01->g * ( 1 - tx ) * ty + c11->g * tx * ty );
        result.r = static_cast<uint8_t> ( c00->r * ( 1 - tx ) * ( 1 - ty ) + c10->r * tx * ( 1 - ty ) + c01->r * ( 1 - tx ) * ty + c11->r * tx * ty );
        result.a = static_cast<uint8_t> ( c00->a * ( 1 - tx ) * ( 1 - ty ) + c10->a * tx * ( 1 - ty ) + c01->a * ( 1 - tx ) * ty + c11->a * tx * ty );
        return result;
    }

    static Color Tile2DInterpolation ( int32_t x, int32_t xstep, float xratio, int32_t y, int32_t ystep, float yratio, int32_t w, int32_t h, int32_t pitch, const Color* buffer )
    {
        int32_t ix = x + ( xstep % w );
        int32_t iy = y + ( ystep % h );
        return buffer[ ( iy * pitch ) + ix];
    }

    static Color NearestNeighbor2DInterpolation ( int32_t x, int32_t xstep, float xratio, int32_t y, int32_t ystep, float yratio, int32_t w, int32_t h, int32_t pitch, const Color* buffer )
    {

        int32_t ix = static_cast<int32_t> ( floorf ( x + ( ( ( xstep + 0.5f ) * xratio ) - 0.5f ) + 0.5f ) );
        int32_t iy = static_cast<int32_t> ( floorf ( y + ( ( ( ystep + 0.5f ) * yratio ) - 0.5f ) + 0.5f ) );

        if ( ix > ( x + w - 1 ) )
        {
            ix = ( x + w - 1 );
        }
        else if ( ix < x )
        {
            ix = x;
        }

        if ( iy > ( y + h - 1 ) )
        {
            iy = ( y + h - 1 );
        }
        else if ( iy < y )
        {
            iy = y;
        }

        return buffer[ ( iy * pitch ) + ix];
    }

    static Color ( *OneDInterpolationFunctions[] ) ( int32_t, int32_t, float, const Color*, int32_t, uint32_t ) =
    {
        NearestNeighbor1DInterpolation, // NEAREST == 0
        Lanczos1DInterpolation,         // LANCZOS == 1
        Tile1DInterpolation,            // TILE    == 2
        LinearInterpolation,          // LINEAR  == 3
    };

    static Color ( *TwoDInterpolationFunctions[] ) ( int32_t, int32_t, float, int32_t, int32_t, float, int32_t, int32_t, int32_t, const Color* ) =
    {
        NearestNeighbor2DInterpolation,  // NEAREST == 0
        Lanczos2DInterpolation,          // LANCZOS == 1
        Tile2DInterpolation,             // TILE == 2
        BilinearInterpolation            // LINEAR  == 3
    };

    void Renderer::DrawSubImage ( const Image* image, int32_t x, int32_t y, int32_t subx, int32_t suby, int32_t subw, int32_t subh, int32_t w, int32_t h, ResizeAlgorithm algorithm )
    {
        assert ( image != NULL );

        uint32_t image_w = image->GetWidth();
        uint32_t image_h = image->GetHeight();

        if ( ( image == NULL ) || ( image_w == 0 ) || ( image_h == 0 ) )
        {
            // Invalid Image
            return;
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
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( OneDInterpolationFunctions[algorithm] ( suby , stepy , ratio_h, image_bitmap + ( ix ), subh, image_w ) );
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
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( OneDInterpolationFunctions[algorithm] ( subx , stepx , ratio_w , image_bitmap + ( iy * image_w ), subw, 1 ) );
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
                            pixels[ ( ( sy * screen_w ) + sx )].Blend ( TwoDInterpolationFunctions[algorithm] ( subx , stepx , ratio_w , suby , stepy , ratio_h , subw, subh, image_w, image_bitmap ) );
                        }
                    }
                }
            }
        }
    }
#endif

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
