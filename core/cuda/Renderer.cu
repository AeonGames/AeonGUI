#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Renderer.h"
namespace AeonGUI
{
    __global__ void blend ( Color* src, size_t src_pitch, uint32_t src_width, uint32_t src_height, Color* dst, size_t dst_pitch, uint32_t dst_width, uint32_t dst_height )
    {
        dst[ ( blockIdx.x * blockDim.x ) + threadIdx.x] = src[ ( blockIdx.x * blockDim.x ) + threadIdx.x];
    }

    void Renderer::DrawSubImage ( Image* image, int32_t x, int32_t y, int32_t subx, int32_t suby, int32_t subw, int32_t subh, int32_t w, int32_t h, ResizeAlgorithm algorithm )
    {
        assert ( image != NULL );

        uint32_t image_w = image->GetWidth();
        uint32_t image_h = image->GetHeight();

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

        Color* d_screen_rect;
        size_t d_screen_pitch;
        Color* d_image_rect;
        size_t d_image_pitch;

        if ( ( w == subw ) && ( h == subh ) )
        {
            // Same scale as original
            cudaMallocPitch ( &d_screen_rect, &d_screen_pitch, w * sizeof ( Color ), h );
            cudaMemcpy2D ( d_screen_rect, d_screen_pitch, pixels, screen_w * sizeof ( Color ), w, h, cudaMemcpyHostToDevice );
            cudaMallocPitch ( &d_image_rect, &d_image_pitch, image_w * sizeof ( Color ), image_h );
            cudaMemcpy2D ( d_image_rect, d_image_pitch, image_bitmap, image_w * sizeof ( Color ), image_w, image_h, cudaMemcpyHostToDevice );
            blend <<< h, w>>> ( d_image_rect, d_image_pitch, image_w, image_h, d_screen_rect, d_screen_pitch, w, h );
#if 0
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
#endif
            cudaMemcpy2D ( pixels, screen_w * sizeof ( Color ), d_screen_rect, d_screen_pitch, w, h, cudaMemcpyDeviceToHost );
            cudaFree ( d_screen_rect );
            cudaFree ( d_image_rect );
        }
#if 0
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
                            //pixels[ ( ( sy * screen_w ) + sx )].Blend ( OneDInterpolationFunctions[algorithm] ( suby , stepy , ratio_h, image_bitmap + ( ix ), subh, image_w ) );
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
                            //pixels[ ( ( sy * screen_w ) + sx )].Blend ( OneDInterpolationFunctions[algorithm] ( subx , stepx , ratio_w , image_bitmap + ( iy * image_w ), subw, 1 ) );
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
                            //pixels[ ( ( sy * screen_w ) + sx )].Blend ( TwoDInterpolationFunctions[algorithm] ( subx , stepx , ratio_w , suby , stepy , ratio_h , subw, subh, image_w, image_bitmap ) );
                        }
                    }
                }
            }
        }
#endif
    }
}
