/*
Copyright (C) 2010-2012,2019 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef AEONGUI_IMAGE_H
#define AEONGUI_IMAGE_H
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include "aeongui/Platform.h"
#include "Integer.h"
#include "Color.h"

namespace AeonGUI
{
    /*! \brief Generic Image class.
    */
    class DLL Image
    {
    public:

        /// Alignment for image content.
        enum Alignment
        {
            LEFT = 0,
            RIGHT,
            CENTER
        };

        /// Type or size of each color element.
        enum Type
        {
            BYTE // one byte per color component
        };

        /// Image data color element order.
        enum Format
        {
            RGB,
            BGR,
            RGBA,
            BGRA
        };

        Image();

        /*!
        \brief Loads a raw image from memory.
        \param image_width The width of the image in memory.
        \param image_height The height of the image in memory.
        \param format The format of the image in memory, RGB,BGR,RGBA or BGRA.
        \param type The type of each color component, currently only BYTE.
        \param data Pointer to the memory buffer containing the image data.
        \todo is the type param really necesary? it's name does not really reflects what it does.
        \return true on success, false otherwise.
        */
        bool Load ( uint32_t image_width, uint32_t image_height, Image::Format format, Image::Type type, const void* data );

        /*!
        \brief Loads an image from a file.
        The function tries to guess the file format based on the file or memory magic number,
        a bare build of the library (no external dependencies), will only look for and process PCX images.
        \param filename Path to the file to load.
        \return true on success, false otherwise.
        \sa Image::LoadFromFile
        */
        bool LoadFromFile ( const char* filename );

        /*!
        \brief Loads an image from a file in memory.
        The function tries to guess the file format based on the file or memory magic number,
        a bare build of the library (no external dependencies), will only look for and process PCX images.
        \param buffer_size Size in bytes of the provided buffer.
        \param buffer Pointer to the buffer containing image data.
        \return true on success, false otherwise.
        \sa Image::Load
        */
        bool LoadFromMemory ( uint32_t buffer_size, void* buffer );

        /*! \brief Unloads and releases image data from memory.*/
        void Unload (  );

        ~Image();

        /*!
        \brief Get the width for the loaded image.
        If no image data is loaded, the returned value will be zero.
        \return Image width, including guides for 9 patch images.
        */
        int32_t GetWidth() const;

        /*!
        \brief Get the height for the loaded image.
        If no image data is loaded, the returned value will be zero.
        \return Image height, including guides for 9 patch images.
        */
        int32_t GetHeight() const;

        /*!
        \brief Retrieve read only bitmap buffer for the image.
        If no image data is loaded, the pointer returned will be NULL.
        \return pointer to the color bitmap buffer for the image object or NULL.
        */
        const Color* GetBitmap() const;

        /*!
        \brief Retrieve Starting X coordinate for patch 9 stretch area.
        \return The starting X coordinate for patch 9 stretching.
        */
        uint32_t GetStretchXStart() const;
        /*!
        \brief Retrieve Starting Y coordinate for patch 9 stretch area.
        \return The starting Y coordinate for patch 9 stretching.
        */
        uint32_t GetStretchYStart() const;
        /*!
        \brief Retrieve Starting X coordinate for patch 9 pad area.
        \return The starting X coordinate for patch 9 padding.
        */
        uint32_t GetPadXStart ( uint32_t width = 0 ) const;
        /*!
        \brief Retrieve Starting Y coordinate for patch 9 pad area.
        \return The starting Y coordinate for patch 9 padding.
        */
        uint32_t GetPadYStart ( uint32_t height = 0 ) const;

        /*!
        \brief Retrieve Ending X coordinate for patch 9 stretch area.
        \return The ending X coordinate for patch 9 stretching.
        */
        uint32_t GetStretchXEnd ( uint32_t width = 0 ) const;
        /*!
        \brief Retrieve Ending Y coordinate for patch 9 stretch area.
        \return The ending Y coordinate for patch 9 stretching.
        */
        uint32_t GetStretchYEnd ( uint32_t height = 0 ) const;
        /*!
        \brief Retrieve Ending X coordinate for patch 9 pad area.
        \return The ending X coordinate for patch 9 padding.
        */
        uint32_t GetPadXEnd ( uint32_t width = 0 ) const;
        /*!
        \brief Retrieve Ending Y coordinate for patch 9 pad area.
        \return The ending Y coordinate for patch 9 padding.
        */
        uint32_t GetPadYEnd ( uint32_t height = 0 ) const;

        /*!
        \brief Retrieve Width of patch 9 stretch area.
        \return The ending X coordinate for patch 9 stretching.
        */
        uint32_t GetStretchWidth ( uint32_t width = 0 ) const;
        /*!
        \brief Retrieve Height of patch 9 stretch area.
        \return The ending Y coordinate for patch 9 stretching.
        */
        uint32_t GetStretchHeight ( uint32_t height = 0 ) const;
        /*!
        \brief Retrieve Width of patch 9 pad area.
        \return The ending X coordinate for patch 9 padding.
        */
        uint32_t GetPadWidth ( uint32_t width = 0 ) const;
        /*!
        \brief Retrieve Height of patch 9 pad area.
        \return The ending Y coordinate for patch 9 padding.
        */
        uint32_t GetPadHeight ( uint32_t height = 0 ) const;

        /*!
        \brief Retrieve raster coordinates to draw image content with the provided dimensions inside the image padding area.
        \param halign [in] Horizontal alignment.
        \param width [in] The width of the extra content to be drawn inside the image padding area.
        \param height [in] The height of the extra content to be drawn inside the image padding area.
        \param valign [in] Vertical alignment.
        \param x [out] Reference to the variable to receive the x coordinate for drawing.
        \param y [out] Reference to the variable to receive the y coordinate for drawing.
        */
        void GetCoordsForDimensions ( Alignment halign, uint32_t width, Alignment valign, uint32_t height, int32_t& x, int32_t& y, uint32_t drawwidth = 0, uint32_t drawheight = 0 ) const;

        /*!
        \brief Retrieve raster X coordinate to draw image content with the provided width inside the image padding area.
        \param halign [in] Horizontal alignment.
        \param width [in] The width of the extra content to be drawn inside the image padding area.
        \return The X coordinate for drawing.
        */
        int32_t GetXCoordForWidth ( Alignment halign, uint32_t width, uint32_t drawwidth = 0 ) const;

        /*!
        \brief Retrieve raster Y coordinate to draw image content with the provided width inside the image padding area.
        \param halign [in] Horizontal alignment.
        \param width [in] The width of the extra content to be drawn inside the image padding area.
        \return The Y coordinate for drawing.
        */
        int32_t GetYCoordForHeight ( Alignment valign, uint32_t height, uint32_t drawheight = 0 ) const;

    private:
        uint32_t width;
        uint32_t height;
        uint32_t stretchxstart;     ///< Patch 9 start stretch coordinate.
        uint32_t stretchxend; ///< Patch 9 end stretch coordinate.
        uint32_t padxstart;         ///< Patch 9 start fill coordinate.
        uint32_t padxend;     ///< Patch 9 end fill coordinate.
        uint32_t stretchystart;     ///< Patch 9 start stretch coordinate.
        uint32_t stretchyend;///< Patch 9 end stretch coordinate.
        uint32_t padystart;         ///< Patch 9 start fill coordinate.
        uint32_t padyend;    ///< Patch 9 end fill coordinate.
        Color* bitmap;
    };
}
#endif
