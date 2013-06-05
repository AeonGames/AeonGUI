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

    bool Image::Load ( uint32_t image_width, uint32_t image_height, Image::Format format, Image::Type type, void* data )
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
                bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 0];
                bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 1];
                bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 2];
                bitmap[i].a = 255;
            }
            break;
        case BGR:
            for ( uint32_t i = 0; i < ( width * height ); ++i )
            {
                bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 0];
                bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 1];
                bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 2];
                bitmap[i].a = 255;
            }
            break;
        case RGBA:
            for ( uint32_t i = 0; i < ( width * height ); ++i )
            {
                bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 0];
                bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 1];
                bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 2];
                bitmap[i].a = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 3];
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
}
