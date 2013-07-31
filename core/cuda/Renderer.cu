#if defined(__GNUC__) && defined(__GNUC_MINOR__) 
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 7)
// This is a hack to make CUDA toolkit 5.0.x work with GCC 4.7.
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif
#endif
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <math_functions.h>
#include "Renderer.h"
namespace AeonGUI
{
    __global__ void blend ( Color* src, size_t src_pitch, uint32_t src_width, uint32_t src_height, Color* dst, size_t dst_pitch, uint32_t dst_width, uint32_t dst_height )
    {
        int32_t x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int32_t y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
        // Avoid accessing out of bounds memory.
        if ( ( x > src_width ) || ( y > src_height ) )
        {
            return;
        }

        // Read values into local variables
        Color dst_color = * ( ( Color* ) ( ( char* ) dst + y * dst_pitch ) + x );
        Color src_color = * ( ( Color* ) ( ( char* ) src + y * src_pitch ) + x );

        if ( ( src_color.a == 255 ) )
        {
            /*  Full source opacity
                do a simple replacement*/
            dst_color.bgra = src_color.bgra;
        }
        /*  If the source alpha is 0
            the destination color is unchanged */
        else if ( src_color.a > 0 )
        {
            float sfactor = ( static_cast<float> ( src_color.a ) / 255.0f );
            float dfactor = 1.0f - sfactor;
            dst_color.r = static_cast<uint8_t> ( min ( 255.0f, ( static_cast<float> ( src_color.r ) * sfactor + static_cast<float> ( dst_color.r ) * dfactor ) ) );
            dst_color.g = static_cast<uint8_t> ( min ( 255.0f, ( static_cast<float> ( src_color.g ) * sfactor + static_cast<float> ( dst_color.g ) * dfactor ) ) );
            dst_color.b = static_cast<uint8_t> ( min ( 255.0f, ( static_cast<float> ( src_color.b ) * sfactor + static_cast<float> ( dst_color.b ) * dfactor ) ) );
            // Just acumulate alpha, if something looks odd try multipling the second addend by dfactor.
            dst_color.a = static_cast<uint8_t> ( min ( 255.0f, ( static_cast<float> ( src_color.a ) + static_cast<float> ( dst_color.a ) ) ) );
        }
        // Write back to destination
        * ( ( Color* ) ( ( char* ) dst + y * dst_pitch ) + x ) = dst_color;
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
            cudaMemcpy2D ( d_screen_rect, d_screen_pitch, pixels + ( ( y * screen_w ) + x ), screen_w * sizeof ( Color ), w * sizeof ( Color ), h, cudaMemcpyHostToDevice );
            cudaMallocPitch ( &d_image_rect, &d_image_pitch, image_w * sizeof ( Color ), image_h );
            cudaMemcpy2D ( d_image_rect, d_image_pitch, image_bitmap, image_w * sizeof ( Color ), image_w * sizeof ( Color ), image_h, cudaMemcpyHostToDevice );

            dim3 gridSize ( static_cast<uint32_t> ( ceilf ( static_cast<float> ( w ) / 16.0f ) ), static_cast<uint32_t> ( ceilf ( static_cast<float> ( h ) / 16.0f ) ) );
            blend <<< gridSize, dim3 ( 16, 16 ) >>> ( d_image_rect, d_image_pitch, image_w, image_h,
                    d_screen_rect, d_screen_pitch, w, h );
            cudaDeviceSynchronize();
            cudaError_t  code = cudaGetLastError();
            if ( code != cudaSuccess )
            {
                printf ( "Cuda error -- %s\n", cudaGetErrorString ( code ) );
            }

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
            cudaMemcpy2D ( pixels + ( ( y * screen_w ) + x ), screen_w * sizeof ( Color ), d_screen_rect, d_screen_pitch, w * sizeof ( Color ), h, cudaMemcpyDeviceToHost );
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
