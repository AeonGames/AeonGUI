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
            printf ( "%s Returning %u bytes instead of requested %u because of end of memory\n", __FUNCTION__, real_length, length );
            memset ( data + real_length, 0, length - real_length );
        }
        read_struct->pointer += real_length;
    }
#endif
    Image::Image () :
            width(0),
            height(0),
            bitmap(NULL)
    {
    }

    bool Image::Load ( uint32_t image_width, uint32_t image_height, Image::Format format, Image::Type type, const void* data )
    {
        assert ( data != NULL );
        width = image_width;
        height = image_height;

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
        return true;
    }

    void Image::Unload()
    {
        if(bitmap!=NULL)
        {
            delete [] bitmap;
            width = 0;
            height = 0;
        }
    }

    Image::~Image()
    {
        Unload();
    }

    int32_t Image::GetWidth()
    {
        return width;
    }

    int32_t Image::GetHeight()
    {
        return height;
    }

    const Color* Image::GetBitmap() const
    {
        return bitmap;
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
            printf("Problem opening %s for reading.\n",filename);
            return false;
        }

        file.seekg ( 0, std::ios_base::end );
        buffer_size = static_cast<uint32_t>(file.tellg());
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
        if(reinterpret_cast<uint8_t*>(buffer)[0]==0x0A)
        {
            // Posible PCX file
            Pcx pcx;
            if(!pcx.Decode(buffer_size,buffer))
            {
                return false;
            }
			if(pcx.GetNumBitPlanes()==3)
			{
				return Load(pcx.GetWidth(),pcx.GetHeight(),RGB,BYTE,pcx.GetPixels());
			}
			else if(pcx.GetNumBitPlanes()==4)
			{
				return Load(pcx.GetWidth(),pcx.GetHeight(),RGBA,BYTE,pcx.GetPixels());
			}
			// Let the Pcx destructor release its pixel buffer.
        }
#if USE_PNG
        else if ( png_sig_cmp ( reinterpret_cast<png_const_bytep>(buffer), 0, 8 ) == 0 )
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
            png_read_memory_struct read_memory_struct = {reinterpret_cast<uint8_t*>(buffer), reinterpret_cast<uint8_t*>(buffer) + 8, buffer_size};
            png_set_read_fn ( png_ptr, &read_memory_struct, png_read_memory_data );
            //png_init_io ( png_ptr, fp );
            png_set_sig_bytes ( png_ptr, 8 );

            png_read_info ( png_ptr, info_ptr );

            png_uint_32 image_width = png_get_image_width ( png_ptr, info_ptr );
            png_uint_32 image_height = png_get_image_height ( png_ptr, info_ptr );
            png_byte color_type = png_get_color_type ( png_ptr, info_ptr );
            png_byte bit_depth = png_get_bit_depth ( png_ptr, info_ptr );

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

                bool retval = Load(image_width,image_height,(color_type==PNG_COLOR_TYPE_RGB)? RGB : RGBA,BYTE,image_buffer);

                free ( image_buffer);
                free ( row_pointers);
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
