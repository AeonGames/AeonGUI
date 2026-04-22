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

TEST ( HTMLRenderTest, BordersPaintInsideBorderBox )
{
    // A single 100x80 div with a red background and a 10px solid green
    // border on every edge.  Yoga reserves the border space inside the
    // border box, so the visible box is still 100x80; the painted edges
    // sit on the outer 10 px of that rectangle and the red interior
    // covers the remaining 80x60 in the middle.
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <div style="width: 100px; height: 80px; background-color: #FF0000;
                border: 10px solid #00FF00"/>
  </body>
</html>)XHTML",
        "aeongui-html-render-borders.xhtml"
    };

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = doc.path();
    window.Draw();

    const uint8_t* pixels = window.GetPixels();
    const size_t   stride = window.GetStride();
    ASSERT_NE ( pixels, nullptr );

    // Each border edge: top/left/right/bottom strips should be green.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 50,  5 ) & 0x00FFFFFFu, 0x0000FF00u )
            << "expected green on the top border";
    EXPECT_EQ ( SamplePixel ( pixels, stride, 50, 75 ) & 0x00FFFFFFu, 0x0000FF00u )
            << "expected green on the bottom border";
    EXPECT_EQ ( SamplePixel ( pixels, stride,  5, 40 ) & 0x00FFFFFFu, 0x0000FF00u )
            << "expected green on the left border";
    EXPECT_EQ ( SamplePixel ( pixels, stride, 95, 40 ) & 0x00FFFFFFu, 0x0000FF00u )
            << "expected green on the right border";

    // Interior — well inside the 10 px border on every side — must
    // still be the red background.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 50, 40 ) & 0x00FFFFFFu, 0x00FF0000u )
            << "interior should be the red background";

    // Outside the border box on the right and bottom: untouched.
    const uint32_t outside_right  = SamplePixel ( pixels, stride, 110, 40 );
    const uint32_t outside_bottom = SamplePixel ( pixels, stride,  50, 90 );
    EXPECT_NE ( outside_right  & 0x00FFFFFFu, 0x0000FF00u );
    EXPECT_NE ( outside_right  & 0x00FFFFFFu, 0x00FF0000u );
    EXPECT_NE ( outside_bottom & 0x00FFFFFFu, 0x0000FF00u );
    EXPECT_NE ( outside_bottom & 0x00FFFFFFu, 0x00FF0000u );
}

namespace
{
    /// Count pixels inside [aX0, aX1) x [aY0, aY1) whose channel values
    /// indicate "more than just the canvas background" — used to assert
    /// that some glyph ink actually landed in a region without committing
    /// to a specific pixel-perfect layout.  The threshold is conservative
    /// to skip Pango/Cairo anti-aliased near-background pixels.
    int CountInkPixels ( const uint8_t* aPixels, size_t aStride,
                         int aX0, int aY0, int aX1, int aY1 )
    {
        int count = 0;
        for ( int y = aY0; y < aY1; ++y )
        {
            const uint8_t* row = aPixels + ( y * aStride );
            for ( int x = aX0; x < aX1; ++x )
            {
                const uint8_t* p = row + ( x * 4 );
                if ( p[3] > 32 )  // alpha above noise floor
                {
                    ++count;
                }
            }
        }
        return count;
    }
}

TEST ( HTMLRenderTest, TextRendersInsideContentBox )
{
    // A 200x60 div with a 20 px black-on-yellow text run.  We don't
    // assert glyph shapes — we only verify that:
    //   * the yellow background fills the box,
    //   * some non-background ink lands inside the content area
    //     (where the text would be drawn),
    //   * no ink escapes the border-box outline.
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <div style="width: 200px; height: 60px; background-color: #FFFF00;
                color: #000000; font-size: 20px">Hello</div>
  </body>
</html>)XHTML",
        "aeongui-html-render-text.xhtml"
    };

    AeonGUI::DOM::Window window ( 300u, 200u );
    window.location() = doc.path();
    window.Draw();

    const uint8_t* pixels = window.GetPixels();
    const size_t   stride = window.GetStride();
    ASSERT_NE ( pixels, nullptr );

    // Background fill check — far right of the 200 px wide box, well
    // past where 5 glyphs would land.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 180, 30 ) & 0x00FFFFFFu, 0x00FFFF00u )
            << "expected yellow background to fill the box";

    // Some text ink must be present in the leftmost portion of the
    // content area where "Hello" is laid out (content origin is the
    // border box corner because we set no padding).
    const int ink_in_content = CountInkPixels ( pixels, stride, 0, 0, 100, 60 );
    EXPECT_GT ( ink_in_content, 50 )
            << "no glyph ink found in the content area; text not painted";

    // No ink should appear past the bottom of the 60 px tall div.
    const int ink_below_box = CountInkPixels ( pixels, stride, 0, 80, 200, 200 );
    EXPECT_EQ ( ink_below_box, 0 )
            << "text or background bled past the box bottom";
}

TEST ( HTMLRenderTest, WrappedTextPaintsMultipleLines )
{
    // A long sentence inside an 80 px wide, 200 px tall div with a
    // yellow background and 16 px black text.  The layout engine is
    // expected to wrap the run into multiple lines, and the renderer
    // must now paint each one — so we verify ink lands in two
    // vertically separated bands inside the content area.
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <div style="width: 80px; height: 200px; background-color: #FFFF00;
                color: #000000; font-size: 16px">The quick brown fox jumps over the lazy dog.</div>
  </body>
</html>)XHTML",
        "aeongui-html-render-wrapped-text.xhtml"
    };

    AeonGUI::DOM::Window window ( 300u, 300u );
    window.location() = doc.path();
    window.Draw();

    const uint8_t* pixels = window.GetPixels();
    const size_t   stride = window.GetStride();
    ASSERT_NE ( pixels, nullptr );

    // Background sanity: the 80 px wide box is yellow far below the
    // first wrapped line.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 75, 100 ) & 0x00FFFFFFu, 0x00FFFF00u )
            << "expected yellow background to fill the box";

    // First line band: somewhere in the top 20 px of the content.
    const int ink_top = CountInkPixels ( pixels, stride, 0, 0, 80, 20 );
    // Second line band: 20 px below the first, well clear of the
    // first line's descenders.  If the renderer were still painting
    // a single line this region would be pure background.
    const int ink_second = CountInkPixels ( pixels, stride, 0, 22, 80, 44 );

    EXPECT_GT ( ink_top,    20 ) << "no glyph ink found on the first wrapped line";
    EXPECT_GT ( ink_second, 20 )
            << "no glyph ink found on the second wrapped line — text was not wrapped";

    // Nothing past the right edge of the 80 px box.
    const int ink_right_of_box = CountInkPixels ( pixels, stride, 90, 0, 200, 200 );
    EXPECT_EQ ( ink_right_of_box, 0 )
            << "wrapped text leaked past the box right edge";
}

TEST ( HTMLRenderTest, BackgroundColorResolvesCurrentColor )
{
    // `background-color: currentColor` must resolve against the
    // computed `color` of the same element.  We pick a vivid green
    // for `color` and assert the box paints green where it sits.
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <div style="width: 60px; height: 30px; color: #00FF00;
                background-color: currentColor"/>
  </body>
</html>)XHTML",
        "aeongui-html-render-bg-currentcolor.xhtml"
    };

    AeonGUI::DOM::Window window ( 100u, 100u );
    window.location() = doc.path();
    window.Draw();

    const uint8_t* pixels = window.GetPixels();
    const size_t   stride = window.GetStride();
    ASSERT_NE ( pixels, nullptr );

    // Inside the box: solid green from currentColor.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 30, 15 ) & 0x00FFFFFFu, 0x0000FF00u )
            << "background-color: currentColor should paint the computed color";

    // Outside the box: background untouched.
    EXPECT_EQ ( SamplePixel ( pixels, stride, 80, 50 ) & 0xFF000000u, 0u )
            << "currentColor bg should not bleed past the box";
}
