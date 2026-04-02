/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_RASTERIMAGE_H
#define AEONGUI_RASTERIMAGE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    /** @brief Decoded raster image.
     *
     *  Loads PNG, JPEG, or PCX images from files or memory and
     *  provides access to the decoded RGBA pixel data.
     */
    class RasterImage
    {
    public:
        /** @brief Supported encoded image formats. */
        enum class EncodedFormat
        {
            Unknown, ///< Format not recognized.
            PNG,     ///< PNG format.
            JPEG,    ///< JPEG format.
            PCX      ///< PCX format.
        };

        /** @brief Decoded pixel formats. */
        enum class PixelFormat
        {
            Unknown, ///< No decoded data.
            RGBA8    ///< 8 bits per channel, RGBA.
        };

        /** @brief Default constructor. Creates an empty (unloaded) image. */
        DLL RasterImage();

        /** @brief Load an image from a file.
         *  @param aPath Path to the image file.
         *  @throws std::runtime_error if loading fails.
         */
        DLL void LoadFromFile ( const std::string& aPath );
        /** @brief Load an image from a memory buffer.
         *  @param aData Pointer to the encoded image data.
         *  @param aSize Size of the data in bytes.
         *  @throws std::runtime_error if loading fails.
         */
        DLL void LoadFromMemory ( const void* aData, size_t aSize );
        /** @brief Release the decoded pixel data. */
        DLL void Clear();

        /** @brief Check whether an image has been loaded.
         *  @return true if pixel data is available.
         */
        [[nodiscard]] DLL bool IsLoaded() const;
        /** @brief Get the original encoded format of the loaded image.
         *  @return The encoded format.
         */
        [[nodiscard]] DLL EncodedFormat GetEncodedFormat() const;
        /** @brief Get the decoded pixel format.
         *  @return The pixel format.
         */
        [[nodiscard]] DLL PixelFormat GetPixelFormat() const;
        /** @brief Get the image width in pixels.
         *  @return Width in pixels.
         */
        [[nodiscard]] DLL uint32_t GetWidth() const;
        /** @brief Get the image height in pixels.
         *  @return Height in pixels.
         */
        [[nodiscard]] DLL uint32_t GetHeight() const;
        /** @brief Get the stride (bytes per row) of the decoded image.
         *  @return Stride in bytes.
         */
        [[nodiscard]] DLL size_t GetStride() const;
        /** @brief Get a pointer to the decoded pixel data.
         *  @return Pointer to RGBA8 pixel data, or nullptr if not loaded.
         */
        [[nodiscard]] DLL const uint8_t* GetPixels() const;
        /** @brief Get a reference to the decoded pixel data vector.
         *  @return Const reference to the internal pixel buffer.
         */
        [[nodiscard]] DLL const std::vector<uint8_t>& GetPixelData() const;

    private:
        EncodedFormat mEncodedFormat;
        PixelFormat mPixelFormat;
        uint32_t mWidth;
        uint32_t mHeight;
        std::vector<uint8_t> mPixelData;
    };
}

#endif
