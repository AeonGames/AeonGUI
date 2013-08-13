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
#include "Image.h"
#include <fstream>
#include "pcx.h"

#ifdef USE_PNG
#include "png.h"
#include <algorithm>
#endif

namespace AeonGUI
{
#ifdef USE_PNG
    struct png_read_memory_struct
    {
        uint8_t* buffer;
        uint8_t* pointer;
        png_size_t size;
    };
    static void png_read_memory_data ( png_structp png_ptr, png_bytep data, png_size_t length )
    {
        if ( png_ptr == NULL )
        {
            printf ( "%s got NULL png_ptr pointer.\n", __FUNCTION__ );
            png_warning ( png_ptr, "Got NULL png_ptr pointer." );
            return;
        }
        png_read_memory_struct* read_struct = static_cast<png_read_memory_struct*> ( png_get_io_ptr ( png_ptr ) );
        // Clip lenght not to get passed the end of the buffer.
        png_size_t real_length = std::min<png_size_t> ( ( ( read_struct->buffer + read_struct->size ) - read_struct->pointer ), length );
        if ( length < 1 )
        {
            printf ( "%s tried to read past end of file\n", __FUNCTION__ );
            png_warning ( png_ptr, "Tried to read past end of file" );
            return;
        }
        memcpy ( data, read_struct->pointer, real_length );
        if ( real_length < length )
        {
            printf ( "%s Returning %lu bytes instead of requested %lu because of end of memory\n", __FUNCTION__, real_length, length );
            memset ( data + real_length, 0, length - real_length );
        }
        read_struct->pointer += real_length;
    }
#endif
    Image::Image () :
        width ( 0 ),
        height ( 0 ),
        stretchxstart ( 0 ),
        stretchwidth ( 0 ),
        padxstart ( 0 ),
        padwidth ( 0 ),
        stretchystart ( 0 ),
        stretchheight ( 0 ),
        padystart ( 0 ),
        padheight ( 0 ),
        bitmap ( NULL )
    {
    }


    static bool GetPatch9DimensionsFromFrame ( const uint8_t* buffer, uint32_t length, uint32_t pitch, Image::Format format, uint32_t& start, uint32_t& end )
    {
        const uint32_t bpp = ( format == Image::RGB || format == Image::BGR ) ? 3 : 4;
        const uint8_t* bytes = buffer;

        start = 0;
        end = 0;

        // First and Last Pixel MUST be white with 0 alpha
        if ( ! ( ( bytes[0] == bytes[1] ) && ( bytes[1] == bytes[2] ) &&  ( bytes[2] == 255 ) && ( ( bpp == 4 ) ? bytes[3] == 0 : true ) ) )
        {
            return false;
        }
        bytes = ( buffer ) + ( ( length - 1 ) * pitch * bpp );
        if ( ! ( ( bytes[0] == bytes[1] ) && ( bytes[1] == bytes[2] ) &&  ( bytes[2] == 255 ) && ( ( bpp == 4 ) ? bytes[3] == 0 : true ) ) )
        {
            return false;
        }

        for ( uint32_t i = 0; i < length; ++i )
        {
            bytes = ( buffer ) + ( i * pitch * bpp );

            if ( ( ( bytes[0] == bytes[1] ) && ( bytes[1] == bytes[2] ) && ( bytes[2] == 0 ) && ( ( bpp == 4 ) ? bytes[3] == 255 : true ) ) )
            {
                // IF pixel is black with 255 alpha
                if ( start == 0 )
                {
                    start = i;
                }
                else if ( end > 0 )
                {
                    // Black line restarts, not a patch 9 frame.
                    start = 0;
                    end = 0;
                    return false;
                }
            }
            else if ( ( ( bytes[0] == bytes[1] ) && ( bytes[1] == bytes[2] ) && ( bytes[2] == 255 ) && ( ( bpp == 4 ) ? bytes[3] == 0 : true ) ) )
            {
                // IF pixel is white with 0 alpha
                if ( ( start > 0 ) && ( end == 0 ) )
                {
                    // Mark the end
                    end = i;
                }
            }
            else
            {
                start = 0;
                end = 0;
                return false;
            }
        }
        if ( start == 0 && end == 0 )
        {
            // All white line, which is ok for Pad.
            return false;
        }
        return true;
    }

    bool Image::Load ( uint32_t image_width, uint32_t image_height, Image::Format format, Image::Type type, const void* data )
    {
        assert ( data != NULL );
        if ( bitmap != NULL )
        {
            Unload();
        }
        // Determine patch9 stretch and pad if any
        bool haspatch9frame = true;

        if ( ( stretchxstart == 0 ) && ( stretchystart == 0 ) && ( stretchwidth == 0 ) && ( stretchheight == 0 ) && ( padxstart == 0 ) && ( padystart == 0 ) && ( padwidth == 0 ) && ( padheight == 0 ) )
        {
            // Stretch values are mandatory
            haspatch9frame = GetPatch9DimensionsFromFrame ( reinterpret_cast<const uint8_t*> ( data ), image_width, 1, format, stretchxstart, stretchwidth );
            if ( haspatch9frame )
            {
                haspatch9frame = GetPatch9DimensionsFromFrame ( reinterpret_cast<const uint8_t*> ( data ), image_height, image_width, format, stretchystart, stretchheight );
            }
            if ( haspatch9frame )
            {
                // Pad values are optional (but the frame must still exist)
                uint32_t bpp = ( format == RGB || format == BGR ) ? 3 : 4;
                GetPatch9DimensionsFromFrame ( reinterpret_cast<const uint8_t*> ( data ) + ( ( image_width * ( image_height - 1 ) ) *bpp ), image_width, 1, format, padxstart, padwidth );
                GetPatch9DimensionsFromFrame ( reinterpret_cast<const uint8_t*> ( data ) + ( ( image_width - 1 ) *bpp ), image_height, image_width, format, padystart, padheight );
            }
        }

        if ( haspatch9frame )
        {
            // Adjust stretch and pad
            stretchxstart -= 1;
            stretchwidth = ( stretchwidth - 1 ) - stretchxstart;
            stretchystart -= 1;
            stretchheight = ( stretchheight - 1 ) - stretchystart;
            padxstart = ( padxstart == 0 ) ? 0 : padxstart - 1;
            padwidth = ( padwidth == 0 ) ? 0 : padwidth - 1;
            padystart = ( padystart == 0 ) ? 0 : padystart - 1;
            padheight = ( padheight == 0 ) ? 0 : padheight - 1;

            // Adjust dimensions
            width = image_width - 2;
            height = image_height - 2;

            bitmap = new Color[width * height];
            switch ( format )
            {
            case RGB:
                for ( uint32_t sy = 1, dy = 0; sy <= height; ++sy, ++dy )
                {
                    for ( uint32_t sx = 1, dx = 0; sx <= width; ++sx, ++dx )
                    {
                        bitmap[ ( dy * width ) + ( dx )].r = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 0];
                        bitmap[ ( dy * width ) + ( dx )].g = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 1];
                        bitmap[ ( dy * width ) + ( dx )].b = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 2];
                        bitmap[ ( dy * width ) + ( dx )].a = 255;
                    }
                }
                break;
            case BGR:
                for ( uint32_t sy = 1, dy = 0; sy <= height; ++sy, ++dy )
                {
                    for ( uint32_t sx = 1, dx = 0; sx <= width; ++sx, ++dx )
                    {
                        bitmap[ ( dy * width ) + ( dx )].b = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 0];
                        bitmap[ ( dy * width ) + ( dx )].g = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 1];
                        bitmap[ ( dy * width ) + ( dx )].r = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 3 ) + ( sx * 3 ) ) + 2];
                        bitmap[ ( dy * width ) + ( dx )].a = 255;
                    }
                }
                break;
            case RGBA:
                for ( uint32_t sy = 1, dy = 0; sy <= height; ++sy, ++dy )
                {
                    for ( uint32_t sx = 1, dx = 0; sx <= width; ++sx, ++dx )
                    {
                        bitmap[ ( dy * width ) + ( dx )].r = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 0];
                        bitmap[ ( dy * width ) + ( dx )].g = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 1];
                        bitmap[ ( dy * width ) + ( dx )].b = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 2];
                        bitmap[ ( dy * width ) + ( dx )].a = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 3];
                    }
                }
                break;
            case BGRA:
                for ( uint32_t sy = 1, dy = 0; sy <= height; ++sy, ++dy )
                {
                    for ( uint32_t sx = 1, dx = 0; sx <= width; ++sx, ++dx )
                    {
                        bitmap[ ( dy * width ) + ( dx )].b = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 0];
                        bitmap[ ( dy * width ) + ( dx )].g = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 1];
                        bitmap[ ( dy * width ) + ( dx )].r = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 2];
                        bitmap[ ( dy * width ) + ( dx )].a = reinterpret_cast<const uint8_t*> ( data ) [ ( ( sy * image_width * 4 ) + ( sx * 4 ) ) + 3];
                    }
                }
                break;
            }
        }
        else
        {
            stretchwidth = width = image_width;
            stretchheight = height = image_height;

            bitmap = new Color[width * height];

            switch ( format )
            {
            case RGB:
                for ( uint32_t i = 0; i < ( width * height ); ++i )
                {
                    bitmap[i].r = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 0];
                    bitmap[i].g = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 1];
                    bitmap[i].b = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 2];
                    bitmap[i].a = 255;
                }
                break;
            case BGR:
                for ( uint32_t i = 0; i < ( width * height ); ++i )
                {
                    bitmap[i].b = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 0];
                    bitmap[i].g = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 1];
                    bitmap[i].r = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 3 ) + 2];
                    bitmap[i].a = 255;
                }
                break;
            case RGBA:
                for ( uint32_t i = 0; i < ( width * height ); ++i )
                {
                    bitmap[i].r = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 4 ) + 0];
                    bitmap[i].g = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 4 ) + 1];
                    bitmap[i].b = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 4 ) + 2];
                    bitmap[i].a = reinterpret_cast<const uint8_t*> ( data ) [ ( i * 4 ) + 3];
                }
                break;
            case BGRA:
                memcpy ( bitmap, data, sizeof ( Color ) * width * height );
                break;
            }
        }
        return true;
    }

    void Image::Unload()
    {
        if ( bitmap != NULL )
        {
            delete [] bitmap;
            width = 0;
            height = 0;
            stretchxstart = 0;
            stretchwidth = 0;
            padxstart = 0;
            padwidth = 0;
            stretchystart = 0;
            stretchheight = 0;
            padystart = 0;
            padheight = 0;
            bitmap = NULL;
        }
    }

    Image::~Image()
    {
        Unload();
    }

    int32_t Image::GetWidth() const
    {
        return width;
    }

    int32_t Image::GetHeight() const
    {
        return height;
    }

    const Color* Image::GetBitmap() const
    {
        return bitmap;
    }

    uint32_t Image::GetStretchXStart() const
    {
        return stretchxstart;
    }

    uint32_t Image::GetStretchYStart() const
    {
        return stretchystart;
    }

    uint32_t Image::GetPadXStart ( uint32_t width ) const
    {
        return padxstart;
    }

    uint32_t Image::GetPadYStart ( uint32_t height ) const
    {
        return padystart;
    }

    uint32_t Image::GetStretchXEnd ( uint32_t width ) const
    {
        return stretchxstart + stretchwidth;
    }

    uint32_t Image::GetStretchYEnd ( uint32_t height ) const
    {
        return stretchystart + stretchheight;
    }

    uint32_t Image::GetPadXEnd ( uint32_t width ) const
    {
        return padxstart + padwidth;
    }

    uint32_t Image::GetPadYEnd ( uint32_t height ) const
    {
        return padystart + padheight;
    }

    uint32_t Image::GetStretchWidth ( uint32_t width ) const
    {
        return stretchwidth;
    }

    uint32_t Image::GetStretchHeight ( uint32_t height ) const
    {
        return stretchheight;
    }

    uint32_t Image::GetPadWidth ( uint32_t width ) const
    {
        return padwidth;
    }

    uint32_t Image::GetPadHeight (  uint32_t height ) const
    {
        return padheight;
    }

    bool Image::LoadFromFile ( const char* filename )
    {
        uint8_t* buffer = NULL;
        uint32_t buffer_size = 0;
        bool retval;
        std::ifstream file;

        file.open ( filename, std::ios_base::in | std::ios_base::binary );
        if ( !file.is_open() )
        {
            printf ( "Problem opening %s for reading.\n", filename );
            return false;
        }

        file.seekg ( 0, std::ios_base::end );
        buffer_size = static_cast<uint32_t> ( file.tellg() );
        file.seekg ( 0, std::ios_base::beg );
        buffer = new uint8_t[buffer_size];
        file.read ( reinterpret_cast<char*> ( buffer ), buffer_size );
        file.close();

        retval = LoadFromMemory ( buffer_size, buffer );
        delete[] buffer;
        return retval;
    }

    bool Image::LoadFromMemory ( uint32_t buffer_size, void* buffer )
    {
        if ( reinterpret_cast<uint8_t*> ( buffer ) [0] == 0x0A )
        {
            // Posible PCX file
            Pcx pcx;
            if ( !pcx.Decode ( buffer_size, buffer ) )
            {
                return false;
            }

            // If the patch9 values are embeded into the image, get them.
            stretchxstart = static_cast<int32_t> ( pcx.GetStretchX() );
            stretchwidth = static_cast<int32_t> ( pcx.GetStretchWidth() );
            padxstart = static_cast<int32_t> ( pcx.GetPadX() );
            padwidth = static_cast<int32_t> ( pcx.GetPadWidth() );
            stretchystart = static_cast<int32_t> ( pcx.GetStretchY() );
            stretchheight = static_cast<int32_t> ( pcx.GetStretchHeight() );
            padystart = static_cast<int32_t> ( pcx.GetPadY() );
            padheight = static_cast<int32_t> ( pcx.GetPadHeight() );

            if ( pcx.GetNumBitPlanes() == 3 )
            {
                return Load ( pcx.GetWidth(), pcx.GetHeight(), RGB, BYTE, pcx.GetPixels() );
            }
            else if ( pcx.GetNumBitPlanes() == 4 )
            {
                return Load ( pcx.GetWidth(), pcx.GetHeight(), RGBA, BYTE, pcx.GetPixels() );
            }
            // Let the Pcx destructor release its pixel buffer.
        }
#if USE_PNG
        else if ( png_sig_cmp ( reinterpret_cast<png_const_bytep> ( buffer ), 0, 8 ) == 0 )
        {
            png_structp png_ptr = png_create_read_struct ( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
            if ( png_ptr == NULL )
            {
                printf ( "png_create_read_struct failed\n" );
                return false;
            }
            png_infop info_ptr = png_create_info_struct ( png_ptr );
            if ( info_ptr == NULL )
            {
                printf ( "png_create_info_struct failed\n" );
                return false;
            }
            if ( setjmp ( png_jmpbuf ( png_ptr ) ) )
            {
                printf ( "Error during init_io\n" );
                return false;
            }
            png_read_memory_struct read_memory_struct = {reinterpret_cast<uint8_t*> ( buffer ), reinterpret_cast<uint8_t*> ( buffer ) + 8, buffer_size};
            png_set_read_fn ( png_ptr, &read_memory_struct, png_read_memory_data );
            //png_init_io ( png_ptr, fp );
            png_set_sig_bytes ( png_ptr, 8 );

            png_read_info ( png_ptr, info_ptr );

            png_uint_32 image_width = png_get_image_width ( png_ptr, info_ptr );
            png_uint_32 image_height = png_get_image_height ( png_ptr, info_ptr );
            png_byte color_type = png_get_color_type ( png_ptr, info_ptr );
            //png_byte bit_depth = png_get_bit_depth ( png_ptr, info_ptr );

            if ( ( color_type == PNG_COLOR_TYPE_RGB ) || ( color_type == PNG_COLOR_TYPE_RGBA ) )
            {
                /*int number_of_passes =*/ png_set_interlace_handling ( png_ptr );
                png_read_update_info ( png_ptr, info_ptr );


                /* read file */
                if ( setjmp ( png_jmpbuf ( png_ptr ) ) )
                {
                    printf ( "Error during read_image\n" );
                    return false;
                }

                png_size_t rowbytes = png_get_rowbytes ( png_ptr, info_ptr );
                png_bytep* row_pointers = ( png_bytep* ) malloc ( static_cast<uint32_t> ( sizeof ( png_bytep ) * image_height ) );
                png_bytep image_buffer = ( png_bytep ) malloc ( static_cast<uint32_t> ( rowbytes * image_height ) );
                for ( png_uint_32 y = 0; y < image_height; ++y )
                {
                    row_pointers[y] = image_buffer + ( rowbytes * y );
                }
                png_read_image ( png_ptr, row_pointers );

                bool retval = Load ( image_width, image_height, ( color_type == PNG_COLOR_TYPE_RGB ) ? RGB : RGBA, BYTE, image_buffer );

                free ( image_buffer );
                free ( row_pointers );
                png_destroy_read_struct ( &png_ptr, &info_ptr, ( png_infopp ) 0 );
                return retval;
            }
            else
            {
                printf ( "PNG image color type not supported\n" );
                return false;
            }
        }
#endif
        return false;
    }
}
