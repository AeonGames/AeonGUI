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

#if (defined(USE_HASH_CRC) && defined(USE_HASH_MD5)) || (defined(USE_HASH_CRC) && defined(USE_HASH_SHA1)) || (defined(USE_HASH_MD5) && defined(USE_HASH_SHA1))
#error Must only define one of USE_HASH_CRC,USE_HASH_MD5 or USE_HASH_SHA1
#endif

extern "C" {
#ifdef USE_HASH_CRC
#include "crc/crc.h"
#endif
#ifdef USE_HASH_MD5
#include "md5/md5.h"
#endif
}
#ifdef USE_HASH_SHA1
#include "sha1/sha1.h"
#endif

namespace AeonGUI
{

#if defined(USE_HASH_CRC)
    std::map<uint32_t, Image::ImageData*> Image::Images;
#elif defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
    std::map<std::string, Image::ImageData*> Image::Images;
#else
    uint32_t Image::imageCount;
#endif

    Image::Image ( const std::string& id, uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data )
    {
        assert ( data != NULL );
#ifdef USE_HASH_CRC
        uint32_t hash = crcFast ( static_cast<const unsigned char*> ( data ),
                                  sizeof ( Color ) * width * height );
        std::map<uint32_t, Image::ImageData*>::iterator it;
#endif
#ifdef USE_HASH_MD5
        char hash[33];
        uint8_t binary_hash[16];

        MD5_CTX ctx;

        std::map<std::string, Image::ImageData*>::iterator it;

        MD5_Init ( &ctx );
        MD5_Update ( &ctx, data, sizeof ( Color ) * width * height );
        MD5_Final ( binary_hash, &ctx );

        const char tab[] = {"0123456789abcdef"};
        for ( int i = 16; --i >= 0; )
        {
            hash[i << 1] = tab[ ( binary_hash[i] >> 4 ) & 0xF];
            hash[ ( i << 1 ) + 1] = tab[binary_hash[i] & 0xF];
        }
        hash[32] = 0;
#endif
#ifdef USE_HASH_SHA1
        char hash[41];
        uint8_t binary_hash[20];
        std::map<std::string, Image::ImageData*>::iterator it;
        sha1::calc ( data, sizeof ( Color ) * width * height, binary_hash );
        sha1::toHexString ( binary_hash, hash );
#endif

#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        it = Images.find ( hash );
        if ( it != Images.end() )
        {
            imageData = it->second;
            imageData->refcount++;
            return;
        }
#endif
        imageData = new ImageData;

#if defined(USE_HASH_CRC)
        imageData->hash = hash;
#elif defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        imageData->hash = hash;
#endif
#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        imageData->refcount = 1;
#endif
        imageData->w = width;
        imageData->h = height;
        imageData->f = format;
        imageData->t = type;

        imageData->bitmap = new Color[width * height];

        switch ( imageData->f )
        {
        case RGB:
            for ( uint32_t i = 0; i < ( width * height ); ++i )
            {
                imageData->bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 0];
                imageData->bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 1];
                imageData->bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 2];
                imageData->bitmap[i].a = 255;
            }
            break;
        case BGR:
            for ( uint32_t i = 0; i < ( width * height ); ++i )
            {
                imageData->bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 0];
                imageData->bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 1];
                imageData->bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 3 ) + 2];
                imageData->bitmap[i].a = 255;
            }
            break;
        case RGBA:
            for ( uint32_t i = 0; i < ( width * height ); ++i )
            {
                imageData->bitmap[i].r = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 0];
                imageData->bitmap[i].g = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 1];
                imageData->bitmap[i].b = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 2];
                imageData->bitmap[i].a = reinterpret_cast<uint8_t*> ( data ) [ ( i * 4 ) + 3];
            }
            break;
        case BGRA:
            memcpy ( imageData->bitmap, data, sizeof ( Color ) * width * height );
            break;
        }
#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        Images[hash] = imageData;
#else
        imageCount++;
#endif
    }

    Image::~Image()
    {
        assert ( imageData != NULL );

#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        if ( --imageData->refcount == 0 )
        {
            Images.erase ( imageData->hash );
#endif

            if ( imageData->bitmap != NULL )
            {
                delete[] imageData->bitmap;
            }
            delete imageData;

#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        }
#else
            imageCount--;
#endif

    }

    int32_t Image::GetWidth()
    {
        return imageData->w;
    }

    int32_t Image::GetHeight()
    {
        return imageData->h;
    }

    Image::Format Image::GetFormat()
    {
        return imageData->f;
    }

    Image::Type Image::GetType()
    {
        return imageData->t;
    }

    const Color* Image::GetBitmap() const
    {
        return imageData->bitmap;
    }
}
