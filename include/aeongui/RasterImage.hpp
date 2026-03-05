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
    class RasterImage
    {
    public:
        enum class EncodedFormat
        {
            Unknown,
            PNG,
            JPEG,
            PCX
        };

        enum class PixelFormat
        {
            Unknown,
            RGBA8
        };

        DLL RasterImage();

        DLL bool LoadFromFile ( const std::string& aPath );
        DLL bool LoadFromMemory ( const void* aData, size_t aSize );
        DLL void Clear();

        [[nodiscard]] DLL bool IsLoaded() const;
        [[nodiscard]] DLL EncodedFormat GetEncodedFormat() const;
        [[nodiscard]] DLL PixelFormat GetPixelFormat() const;
        [[nodiscard]] DLL uint32_t GetWidth() const;
        [[nodiscard]] DLL uint32_t GetHeight() const;
        [[nodiscard]] DLL size_t GetStride() const;
        [[nodiscard]] DLL const uint8_t* GetPixels() const;
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
