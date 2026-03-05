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

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "aeongui/RasterImage.hpp"

namespace
{
    static std::vector<uint8_t> Make1x1Pcx ( uint8_t aValue )
    {
        std::vector<uint8_t> bytes ( 128u, 0u );
        bytes[0] = 0x0Au;  // Identifier
        bytes[1] = 5u;     // Version
        bytes[2] = 1u;     // RLE encoding
        bytes[3] = 8u;     // Bits per pixel per plane

        // XStart/YStart/XEnd/YEnd for 1x1 image.
        bytes[4] = 0u;
        bytes[5] = 0u;
        bytes[6] = 0u;
        bytes[7] = 0u;
        bytes[8] = 0u;
        bytes[9] = 0u;
        bytes[10] = 0u;
        bytes[11] = 0u;

        bytes[65] = 1u; // NumBitPlanes
        bytes[66] = 1u; // BytesPerLine LE
        bytes[67] = 0u;

        // Single raw byte payload for one scanline/one plane.
        bytes.push_back ( aValue );
        return bytes;
    }
}

TEST ( RasterImageTest, LoadPcxFromMemory )
{
    AeonGUI::RasterImage image;
    std::vector<uint8_t> data = Make1x1Pcx ( 0x2Au );

    ASSERT_TRUE ( image.LoadFromMemory ( data.data(), data.size() ) );
    EXPECT_TRUE ( image.IsLoaded() );
    EXPECT_EQ ( image.GetEncodedFormat(), AeonGUI::RasterImage::EncodedFormat::PCX );
    EXPECT_EQ ( image.GetPixelFormat(), AeonGUI::RasterImage::PixelFormat::RGBA8 );
    EXPECT_EQ ( image.GetWidth(), 1u );
    EXPECT_EQ ( image.GetHeight(), 1u );
    EXPECT_EQ ( image.GetStride(), 4u );

    const uint8_t* pixels = image.GetPixels();
    ASSERT_NE ( pixels, nullptr );
    EXPECT_EQ ( pixels[0], 0x2Au );
    EXPECT_EQ ( pixels[1], 0x2Au );
    EXPECT_EQ ( pixels[2], 0x2Au );
    EXPECT_EQ ( pixels[3], 0xFFu );
}

TEST ( RasterImageTest, ReplaceImageOnSuccessfulLoad )
{
    AeonGUI::RasterImage image;
    std::vector<uint8_t> first = Make1x1Pcx ( 0x11u );
    std::vector<uint8_t> second = Make1x1Pcx ( 0x66u );

    ASSERT_TRUE ( image.LoadFromMemory ( first.data(), first.size() ) );
    ASSERT_TRUE ( image.LoadFromMemory ( second.data(), second.size() ) );

    const uint8_t* pixels = image.GetPixels();
    ASSERT_NE ( pixels, nullptr );
    EXPECT_EQ ( pixels[0], 0x66u );
    EXPECT_EQ ( pixels[1], 0x66u );
    EXPECT_EQ ( pixels[2], 0x66u );
    EXPECT_EQ ( pixels[3], 0xFFu );
}

TEST ( RasterImageTest, FailedLoadKeepsPreviousImage )
{
    AeonGUI::RasterImage image;
    std::vector<uint8_t> valid = Make1x1Pcx ( 0x33u );
    std::vector<uint8_t> invalid{0x00u, 0x01u, 0x02u, 0x03u};

    ASSERT_TRUE ( image.LoadFromMemory ( valid.data(), valid.size() ) );
    ASSERT_FALSE ( image.LoadFromMemory ( invalid.data(), invalid.size() ) );

    EXPECT_TRUE ( image.IsLoaded() );
    EXPECT_EQ ( image.GetEncodedFormat(), AeonGUI::RasterImage::EncodedFormat::PCX );
    const uint8_t* pixels = image.GetPixels();
    ASSERT_NE ( pixels, nullptr );
    EXPECT_EQ ( pixels[0], 0x33u );
}
