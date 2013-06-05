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

        Image();

        /*!
        \brief Loads an image from memory.
        \return true on success, false otherwise.
        */
        bool Load ( const std::string& id, uint32_t image_width, uint32_t image_height, Image::Format format, Image::Type type, void* data );

        /*! \brief Unloads and releases image data from memory.*/
        void Unload (  );

        ~Image();

        /*!
        \brief Get the width for the loaded image.
        If no image data is loaded, the returned value will be zero.
        \return Image width, including guides for 9 patch images.
        */
        int32_t GetWidth();

        /*!
        \brief Get the height for the loaded image.
        If no image data is loaded, the returned value will be zero.
        \return Image height, including guides for 9 patch images.
        */
        int32_t GetHeight();

        /*!
        \brief Retrieve read only bitmap buffer for the image.
        If no image data is loaded, the pointer returned will be NULL.
        \return pointer to the color bitmap buffer for the image object or NULL.
        */
        const Color* GetBitmap() const;

    private:
        uint32_t width;
        uint32_t height;
        Color* bitmap;
    };
}
#endif
