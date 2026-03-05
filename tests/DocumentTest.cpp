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
