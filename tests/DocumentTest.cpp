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
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/SVGGeometryElement.hpp"
#include "aeongui/Canvas.hpp"

namespace
{
    static std::vector<uint8_t> Make1x1Pcx ( uint8_t aValue )
    {
        std::vector<uint8_t> bytes ( 128u, 0u );
        bytes[0] = 0x0Au;
        bytes[1] = 5u;
        bytes[2] = 1u;
        bytes[3] = 8u;

        bytes[4] = 0u;
        bytes[5] = 0u;
        bytes[6] = 0u;
        bytes[7] = 0u;
        bytes[8] = 0u;
        bytes[9] = 0u;
        bytes[10] = 0u;
        bytes[11] = 0u;

        bytes[65] = 1u;
        bytes[66] = 1u;
        bytes[67] = 0u;

        bytes.push_back ( aValue );
        return bytes;
    }
}

TEST ( DocumentTest, StoresLoadedUrl )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-document-test.svg";

    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"></svg>";
    }

    AeonGUI::DOM::Document document;
    EXPECT_NO_THROW ( document.Load ( tempPath.string() ) );

    // Document::Load normalizes system paths to file:// URLs
    std::string generic = std::filesystem::absolute ( tempPath ).generic_string();
    if ( !generic.empty() && generic[0] != '/' )
    {
        generic = "/" + generic;
    }
    EXPECT_EQ ( document.url(), "file://" + generic );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, LoadSvgWithStyleAndClasses )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-class-test.svg";

    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<style>.btn { fill: #39f; } .btn:hover { fill: #5bf; }</style>)"
             << R"(<rect class="btn" x="10" y="10" width="80" height="80"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    EXPECT_NO_THROW ( document.Load ( tempPath.string() ) );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, LoadHoverDemoSvg )
{
    const std::filesystem::path svgPath = std::filesystem::path ( SOURCE_PATH ) / "images" / "hover-demo.svg";
    ASSERT_TRUE ( std::filesystem::exists ( svgPath ) ) << "hover-demo.svg not found at " << svgPath;

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( svgPath.string() ) );

    auto canvas = AeonGUI::Canvas::Create ( 800u, 600u );
    canvas->Clear();
    ASSERT_NO_THROW ( document.Draw ( *canvas ) );
}

TEST ( DocumentTest, DrawResolvesImageHrefWithFragment )
{
    const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "aeongui-document-image-test";
    const std::filesystem::path svgPath = tempDir / "doc.svg";
    const std::filesystem::path imagePath = tempDir / "img.pcx";

    std::error_code ec;
    std::filesystem::create_directories ( tempDir, ec );
    ASSERT_FALSE ( ec );

    {
        const std::vector<uint8_t> image = Make1x1Pcx ( 0x7Fu );
        std::ofstream file ( imagePath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file.write ( reinterpret_cast<const char*> ( image.data() ), static_cast<std::streamsize> ( image.size() ) );
    }

    {
        std::ofstream file ( svgPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"2\" height=\"2\">"
             << "<image x=\"0\" y=\"0\" width=\"1\" height=\"1\" xlink:href=\"img.pcx#thumb\"/>"
             << "</svg>";
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( svgPath.string() ) );

    auto canvas = AeonGUI::Canvas::Create ( 2u, 2u );
    canvas->Clear();
    ASSERT_NO_THROW ( document.Draw ( *canvas ) );

    std::filesystem::remove ( svgPath, ec );
    std::filesystem::remove ( imagePath, ec );
    std::filesystem::remove ( tempDir, ec );
}

TEST ( DocumentTest, SetAttributeUpdatesRectPath )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-setattr-rect.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
             << R"(<rect id="r" x="10" y="10" width="80" height="40"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( tempPath.string() ) );

    AeonGUI::DOM::Element* elem = document.getElementById ( "r" );
    ASSERT_NE ( elem, nullptr );
    auto* geom = dynamic_cast<AeonGUI::DOM::SVGGeometryElement*> ( elem );
    ASSERT_NE ( geom, nullptr );

    double lengthBefore = geom->GetPath().GetTotalLength();
    EXPECT_GT ( lengthBefore, 0.0 );

    elem->setAttribute ( "width", "160" );
    double lengthAfter = geom->GetPath().GetTotalLength();
    EXPECT_GT ( lengthAfter, lengthBefore );

    // getAttribute should reflect the new value.
    const auto* widthAttr = elem->getAttribute ( "width" );
    ASSERT_NE ( widthAttr, nullptr );
    EXPECT_EQ ( *widthAttr, "160" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, SetAttributeUpdatesCirclePath )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-setattr-circle.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
             << R"(<circle id="c" cx="100" cy="100" r="40"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( tempPath.string() ) );

    AeonGUI::DOM::Element* elem = document.getElementById ( "c" );
    ASSERT_NE ( elem, nullptr );
    auto* geom = dynamic_cast<AeonGUI::DOM::SVGGeometryElement*> ( elem );
    ASSERT_NE ( geom, nullptr );

    double lengthBefore = geom->GetPath().GetTotalLength();
    elem->setAttribute ( "r", "80" );
    double lengthAfter = geom->GetPath().GetTotalLength();
    EXPECT_GT ( lengthAfter, lengthBefore );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, SetAttributeUpdatesPathData )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-setattr-path.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
             << R"(<path id="p" d="M 0 0 L 100 0"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( tempPath.string() ) );

    AeonGUI::DOM::Element* elem = document.getElementById ( "p" );
    ASSERT_NE ( elem, nullptr );
    auto* geom = dynamic_cast<AeonGUI::DOM::SVGGeometryElement*> ( elem );
    ASSERT_NE ( geom, nullptr );

    double lengthBefore = geom->GetPath().GetTotalLength();
    EXPECT_NEAR ( lengthBefore, 100.0, 0.5 );

    elem->setAttribute ( "d", "M 0 0 L 200 0" );
    double lengthAfter = geom->GetPath().GetTotalLength();
    EXPECT_NEAR ( lengthAfter, 200.0, 0.5 );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, SetAttributeUpdatesIdLookup )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-setattr-id.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect id="oldid" x="0" y="0" width="50" height="50"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( tempPath.string() ) );

    AeonGUI::DOM::Element* elem = document.getElementById ( "oldid" );
    ASSERT_NE ( elem, nullptr );

    // After setAttribute("id", "newid"), the id() accessor should reflect it.
    elem->setAttribute ( "id", "newid" );
    EXPECT_EQ ( elem->id(), "newid" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorByType )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-type.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect id="r1" x="0" y="0" width="50" height="50"/>)"
             << R"(<circle id="c1" cx="50" cy="50" r="25"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    auto* elem = document.querySelector ( "rect" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->id(), "r1" );

    auto* circle = document.querySelector ( "circle" );
    ASSERT_NE ( circle, nullptr );
    EXPECT_EQ ( circle->id(), "c1" );

    auto* none = document.querySelector ( "line" );
    EXPECT_EQ ( none, nullptr );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorById )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-id.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect id="myRect" x="0" y="0" width="50" height="50"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    auto* elem = document.querySelector ( "#myRect" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->tagName(), "rect" );

    auto* none = document.querySelector ( "#missing" );
    EXPECT_EQ ( none, nullptr );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorByClass )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-class.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect class="btn" id="r1" x="0" y="0" width="50" height="50"/>)"
             << R"(<rect class="label" id="r2" x="0" y="50" width="50" height="50"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    auto* elem = document.querySelector ( ".btn" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->id(), "r1" );

    auto* label = document.querySelector ( ".label" );
    ASSERT_NE ( label, nullptr );
    EXPECT_EQ ( label->id(), "r2" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorCompound )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-compound.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect class="btn" id="r1" x="0" y="0" width="50" height="50"/>)"
             << R"(<circle class="btn" id="c1" cx="50" cy="50" r="25"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    auto* elem = document.querySelector ( "circle.btn" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->id(), "c1" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorDescendant )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-desc.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<g id="group1">)"
             << R"(<rect id="r1" x="0" y="0" width="50" height="50"/>)"
             << R"(</g>)"
             << R"(<rect id="r2" x="50" y="0" width="50" height="50"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    // Descendant combinator: rect inside g
    auto* elem = document.querySelector ( "g rect" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->id(), "r1" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorChild )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-child.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<g id="outer"><g id="inner"><rect id="r1" x="0" y="0" width="50" height="50"/></g></g>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    // Child combinator: svg > g should match "outer"
    auto* elem = document.querySelector ( "svg > g" );
    ASSERT_NE ( elem, nullptr );
    EXPECT_EQ ( elem->id(), "outer" );

    // Descendant should still work through nested groups
    auto* r1 = document.querySelector ( "svg rect" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_EQ ( r1->id(), "r1" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorAll )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qsa.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect id="r1" x="0" y="0" width="25" height="25"/>)"
             << R"(<rect id="r2" x="25" y="0" width="25" height="25"/>)"
             << R"(<circle id="c1" cx="50" cy="50" r="10"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    auto rects = document.querySelectorAll ( "rect" );
    ASSERT_EQ ( rects.size(), 2u );
    EXPECT_EQ ( rects[0]->id(), "r1" );
    EXPECT_EQ ( rects[1]->id(), "r2" );

    auto all = document.querySelectorAll ( "*" );
    // svg + rect + rect + circle = 4 elements
    EXPECT_EQ ( all.size(), 4u );

    auto empty = document.querySelectorAll ( "line" );
    EXPECT_TRUE ( empty.empty() );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, ElementQuerySelector )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-elem-qs.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">)"
             << R"(<g id="btn1"><rect id="r1" x="0" y="0" width="50" height="50"/><text id="t1">OK</text></g>)"
             << R"(<g id="btn2"><rect id="r2" x="60" y="0" width="50" height="50"/><text id="t2">Cancel</text></g>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    // querySelector on an Element should only search within that element's subtree
    auto* btn1 = document.getElementById ( "btn1" );
    ASSERT_NE ( btn1, nullptr );

    auto* rectInBtn1 = btn1->querySelector ( "rect" );
    ASSERT_NE ( rectInBtn1, nullptr );
    EXPECT_EQ ( rectInBtn1->id(), "r1" );

    auto* textInBtn1 = btn1->querySelector ( "text" );
    ASSERT_NE ( textInBtn1, nullptr );
    EXPECT_EQ ( textInBtn1->id(), "t1" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

TEST ( DocumentTest, QuerySelectorCommaList )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "aeongui-qs-comma.svg";
    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
             << R"(<rect id="r1" x="0" y="0" width="50" height="50"/>)"
             << R"(<circle id="c1" cx="75" cy="75" r="10"/>)"
             << R"(</svg>)";
    }

    AeonGUI::DOM::Document document;
    document.Load ( tempPath.string() );

    // Comma-separated selectors: first match in document order
    auto* elem = document.querySelector ( "circle, rect" );
    ASSERT_NE ( elem, nullptr );
    // rect appears first in document order
    EXPECT_EQ ( elem->id(), "r1" );

    auto all = document.querySelectorAll ( "rect, circle" );
    ASSERT_EQ ( all.size(), 2u );
    EXPECT_EQ ( all[0]->id(), "r1" );
    EXPECT_EQ ( all[1]->id(), "c1" );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}
