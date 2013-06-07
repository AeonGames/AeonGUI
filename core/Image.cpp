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
        return false;
    }
}
