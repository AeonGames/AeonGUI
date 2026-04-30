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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "aeongui/Cursor.hpp"
#include "aeongui/RasterImage.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Window.hpp"

namespace
{
    // Build a 1x1 PCX containing a single 8-bit raw value (palette index 0).
    // Mirrors the helper used by RasterImageTest.
    static std::vector<uint8_t> Make1x1Pcx ( uint8_t aValue )
    {
        std::vector<uint8_t> bytes ( 128u, 0u );
        bytes[0] = 0x0Au;
        bytes[1] = 5u;
        bytes[2] = 1u;
        bytes[3] = 8u;
        bytes[65] = 1u;
        bytes[66] = 1u;
        bytes[67] = 0u;
        // PCX uses RLE: any byte with the top two bits set (>= 0xC0) is a
        // count marker, so always emit the explicit count form (count=1)
        // followed by the literal value to avoid misinterpretation for
        // values like 0xFF or 0xC8.
        bytes.push_back ( 0xC1u );
        bytes.push_back ( aValue );
        return bytes;
    }

    // Build a small fully-opaque solid grayscale RasterImage by hand:
    // forge a 1x1 PCX with the given gray @p aValue. The PCX decoder
    // treats 1-bitplane data as grayscale, so the resulting RGBA pixel is
    // (aValue, aValue, aValue, 255). Keeps the test free of file I/O.
    static AeonGUI::RasterImage MakeSolidImage ( uint8_t aValue )
    {
        std::vector<uint8_t> bytes = Make1x1Pcx ( aValue );
        AeonGUI::RasterImage image;
        image.LoadFromMemory ( bytes.data(), bytes.size() );
        return image;
    }

    class TempSVG
    {
    public:
        explicit TempSVG ( const std::string& aSvg, const std::string& aName = "aeongui-cursor.svg" )
            : mPath ( std::filesystem::temp_directory_path() / aName )
        {
            std::ofstream f ( mPath, std::ios::binary | std::ios::out );
            f << aSvg;
        }
        ~TempSVG()
        {
            std::error_code ec;
            std::filesystem::remove ( mPath, ec );
        }
        std::string path() const
        {
            return mPath.generic_string();
        }
    private:
        std::filesystem::path mPath;
    };
}

TEST ( CursorTest, DefaultIsDisabledAndEmpty )
{
    AeonGUI::Cursor cursor;
    EXPECT_FALSE ( cursor.IsEnabled() );
    EXPECT_FALSE ( cursor.HasSource() );
    EXPECT_FALSE ( cursor.HasBackingStore() );
    EXPECT_EQ ( cursor.GetWidth(), 0u );
    EXPECT_EQ ( cursor.GetHeight(), 0u );
}

TEST ( CursorTest, DisabledCompositeIsNoOp )
{
    AeonGUI::Cursor cursor;
    AeonGUI::RasterImage img = MakeSolidImage ( 200u );
    cursor.SetSource ( img, 0, 0 );
    // Note: cursor stays disabled by default.

    std::vector<uint8_t> buffer ( 4u * 4u * 4u, 0xAAu );
    std::vector<uint8_t> snapshot = buffer;
    cursor.Composite ( buffer.data(), 4, 4, 4 * 4, 1, 1 );
    EXPECT_EQ ( buffer, snapshot );
    EXPECT_FALSE ( cursor.HasBackingStore() );
}

TEST ( CursorTest, RasterSetSourcePopulatesMetadata )
{
    AeonGUI::Cursor cursor;
    AeonGUI::RasterImage img = MakeSolidImage ( 10u );
    cursor.SetSource ( img, 3, 5 );
    EXPECT_TRUE ( cursor.HasSource() );
    EXPECT_EQ ( cursor.GetWidth(), img.GetWidth() );
    EXPECT_EQ ( cursor.GetHeight(), img.GetHeight() );
    EXPECT_EQ ( cursor.GetHotspotX(), 3 );
    EXPECT_EQ ( cursor.GetHotspotY(), 5 );
}

TEST ( CursorTest, CompositeRestoreRoundTrip )
{
    AeonGUI::Cursor cursor;
    cursor.SetSource ( MakeSolidImage ( 0xFFu ), 0, 0 );
    cursor.SetEnabled ( true );

    // 4x4 BGRA buffer pre-filled with a recognizable pattern.
    std::vector<uint8_t> buffer ( 4u * 4u * 4u );
    for ( size_t i = 0; i < buffer.size(); ++i )
    {
        buffer[i] = static_cast<uint8_t> ( i );
    }
    std::vector<uint8_t> original = buffer;

    cursor.Composite ( buffer.data(), 4, 4, 4 * 4, 2, 2 );
    EXPECT_TRUE ( cursor.HasBackingStore() );
    // The pixel at (2,2) must now be solid white premultiplied BGRA = (255,255,255,255).
    const uint8_t* p = buffer.data() + ( 2u * 4u + 2u ) * 4u;
    EXPECT_EQ ( p[0], 0xFFu );
    EXPECT_EQ ( p[1], 0xFFu );
    EXPECT_EQ ( p[2], 0xFFu );
    EXPECT_EQ ( p[3], 0xFFu );

    cursor.Restore ( buffer.data(), 4, 4, 4 * 4 );
    EXPECT_FALSE ( cursor.HasBackingStore() );
    EXPECT_EQ ( buffer, original );
}

TEST ( CursorTest, OffscreenCompositeDoesNotWriteOutOfBounds )
{
    AeonGUI::Cursor cursor;
    cursor.SetSource ( MakeSolidImage ( 0xFFu ), 0, 0 );
    cursor.SetEnabled ( true );

    // Allocate an oversized buffer with sentinel borders; the cursor lives
    // on a 4x4 logical surface starting at offset 16 bytes (one row).
    std::vector<uint8_t> buffer ( ( 4u + 2u ) * 4u * 4u, 0x5Au );
    cursor.Composite ( buffer.data() + 4u * 4u, 4, 4, 4 * 4, -10, -10 );

    // Sentinels must remain untouched.
    for ( size_t i = 0; i < 4u * 4u; ++i )
    {
        EXPECT_EQ ( buffer[i], 0x5Au ) << "front sentinel at " << i;
    }
    for ( size_t i = ( 4u + 1u ) * 4u * 4u; i < buffer.size(); ++i )
    {
        EXPECT_EQ ( buffer[i], 0x5Au ) << "trailing sentinel at " << i;
    }
    EXPECT_FALSE ( cursor.HasBackingStore() );
}

TEST ( CursorTest, RejectsUnloadedRasterImage )
{
    AeonGUI::Cursor cursor;
    AeonGUI::RasterImage empty;
    EXPECT_THROW ( cursor.SetSource ( empty, 0, 0 ), std::runtime_error );
}

TEST ( CursorTest, RejectsZeroSizedSvgRender )
{
    AeonGUI::Cursor cursor;
    EXPECT_THROW ( cursor.SetSource ( std::string{"file:///nonexistent"}, 0u, 16u ), std::runtime_error );
    EXPECT_THROW ( cursor.SetSource ( std::string{"file:///nonexistent"}, 16u, 0u ), std::runtime_error );
}

TEST ( CursorTest, SvgSetSourceProducesRequestedSize )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
        <rect x="0" y="0" width="16" height="16" fill="red"/>
    </svg>)SVG"};

    AeonGUI::Cursor cursor;
    ASSERT_NO_THROW ( cursor.SetSource ( svg.path(), 16u, 16u ) );
    EXPECT_EQ ( cursor.GetWidth(), 16u );
    EXPECT_EQ ( cursor.GetHeight(), 16u );
    EXPECT_EQ ( cursor.GetHotspotX(), 0 );
    EXPECT_EQ ( cursor.GetHotspotY(), 0 );
}

TEST ( CursorTest, MouseMoveDoesNotDirtyDocument )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
        <rect id="r" x="0" y="0" width="64" height="64" fill="white"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 64u, 64u );
    window.location() = svg.path();
    EXPECT_TRUE ( window.Draw() );  // initial paint

    window.cursor().SetSource ( MakeSolidImage ( 0xFFu ), 0, 0 );
    window.cursor().SetEnabled ( true );

    // First move sets up the hover state on the rect; CSS reselection may
    // legitimately dirty the document — that is unrelated to the cursor.
    window.HandleMouseMove ( 10.0, 10.0 );
    if ( window.document()->IsDirty() )
    {
        EXPECT_TRUE ( window.Draw() );
    }

    // A subsequent move that stays on the same element must NOT dirty the
    // document: the cursor is composited directly into the canvas buffer
    // without going through the document/draw pipeline.
    window.HandleMouseMove ( 30.0, 30.0 );
    EXPECT_FALSE ( window.document()->IsDirty() );
    EXPECT_FALSE ( window.Draw() );

    window.HandleMouseMove ( 50.0, 50.0 );
    EXPECT_FALSE ( window.document()->IsDirty() );
    EXPECT_FALSE ( window.Draw() );
}
