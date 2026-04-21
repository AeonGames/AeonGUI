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

// Phase 4 verification (slice 1, backgrounds only):
//  * Loading an XHTML document through Window triggers HTML layout
//    via Window::FullDraw and HTMLLayoutEngine::Layout.
//  * HTMLElement::DrawStart paints background-color using the
//    laid-out border box, in document coordinates that match what
//    HTMLLayoutEngineTest verified.
//
// We sample the rendered framebuffer at a few well-known pixels
// rather than hashing the whole image: that way we don't depend on
// the renderer's pixel-perfect output (Cairo vs. Skia), only on
// "the colored rectangle ended up at the right place".

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Location.hpp"

namespace
{
    /// Scoped temporary file that writes the given XHTML content on
    /// construction and removes the file on destruction.
    class TempXHTML
    {
    public:
        explicit TempXHTML ( const std::string& aContent,
                             const std::string& aName = "aeongui-html-render.xhtml" )
            : mPath{ std::filesystem::temp_directory_path() / aName }
        {
            std::ofstream file ( mPath, std::ios::binary | std::ios::out );
            file << aContent;
        }
        ~TempXHTML()
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

    /// Both Cairo and Skia canvases expose pixels as BGRA8.  This is
    /// also the format Color is laid out in.  Returned as 0xAARRGGBB.
    uint32_t SamplePixel ( const uint8_t* aPixels, size_t aStride, int aX, int aY )
    {
        const uint8_t* p = aPixels + ( aY * aStride ) + ( aX * 4 );
        const uint8_t b = p[0];
        const uint8_t g = p[1];
        const uint8_t r = p[2];
        const uint8_t a = p[3];
        return ( static_cast<uint32_t> ( a ) << 24 ) |
               ( static_cast<uint32_t> ( r ) << 16 ) |
               ( static_cast<uint32_t> ( g ) <<  8 ) |
               static_cast<uint32_t> ( b );
    }
}

TEST ( HTMLRenderTest, BackgroundColorFillsBorderBox )
{
    // Two stacked block divs with explicit sizes and background colors.
    // Default flex-direction (column) for non-flex containers means
    // they stack vertically.  At a 200x200 viewport we expect:
    //   * red   block from y=0..40
    //   * blue  block from y=40..100
    //   * background untouched outside those rows
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <div style="width: 100px; height: 40px; background-color: #FF0000"/>
    <div style="width: 50px;  height: 60px; background-color: #0000FF"/>
  </body>
</html>)XHTML"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = doc.path();
    window.Draw();

    const uint8_t* pixels = window.GetPixels();
    const size_t   stride = window.GetStride();
    ASSERT_NE ( pixels, nullptr );

    // Inside the red block.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 50, 20 ) & 0x00FFFFFFu, 0x00FF0000u )
            << "expected red background inside first <div>";

    // Inside the blue block (which starts at y=40 and is only 50px wide).
    EXPECT_EQ ( SamplePixel ( pixels, stride, 25, 70 ) & 0x00FFFFFFu, 0x000000FFu )
            << "expected blue background inside second <div>";

    // To the right of the narrower second block — should be untouched
    // background, not blue.
    const uint32_t outside_blue = SamplePixel ( pixels, stride, 120, 70 );
    EXPECT_NE ( outside_blue & 0x00FFFFFFu, 0x000000FFu )
            << "second <div> bled past its 50px width";

    // Far below either block — should also be untouched.
    const uint32_t below_blocks = SamplePixel ( pixels, stride, 50, 150 );
    EXPECT_NE ( below_blocks & 0x00FFFFFFu, 0x00FF0000u );
    EXPECT_NE ( below_blocks & 0x00FFFFFFu, 0x000000FFu );
}
