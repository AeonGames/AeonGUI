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
#include "aeongui/CairoCanvas.hpp"

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
    EXPECT_EQ ( document.url(), tempPath.string() );

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

    AeonGUI::CairoCanvas canvas ( 800u, 600u );
    canvas.Clear();
    ASSERT_NO_THROW ( document.Draw ( canvas ) );
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

    AeonGUI::CairoCanvas canvas ( 2u, 2u );
    canvas.Clear();
    ASSERT_NO_THROW ( document.Draw ( canvas ) );

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
