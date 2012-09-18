#ifndef AEONGUI_IMAGE_H
#define AEONGUI_IMAGE_H
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
#include <cstddef>
#include <cassert>
#include <cstring>
#include <string>
#include <map>
#include "Integer.h"
#include "Color.h"

namespace AeonGUI
{
    /*! \brief Generic Image class.
        \todo Write some code to avoid multiple copies of the same image in memory
    */
    class Image
    {
    public:

        enum Type
        {
            BYTE // one byte per color component
        };

        enum Format
        {
            RGB,
            BGR,
            RGBA,
            BGRA
        };

        Image ( const std::string& id, uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data );

        ~Image();

        int32_t GetWidth();

        int32_t GetHeight();

        Format GetFormat();

        Type GetType();

        const Color* GetBitmap() const;

    private:

        struct ImageData
        {
#if defined(USE_HASH_CRC)
            uint32_t hash;
#elif defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
            std::string hash;
#endif
#if defined(USE_HASH_CRC) || defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
            uint32_t refcount;
#endif
            int32_t w;
            int32_t h;
            Format f;
            Type t;
            Color* bitmap;
        };

        ImageData* imageData;

#if defined(USE_HASH_CRC)
        static std::map<uint32_t, ImageData*> Images;
#elif defined(USE_HASH_MD5) || defined(USE_HASH_SHA1)
        static std::map<std::string, ImageData*> Images;
#else
        static uint32_t imageCount;
#endif
    };
}
#endif
