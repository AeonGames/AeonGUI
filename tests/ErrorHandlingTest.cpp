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
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include "aeongui/AeonGUI.hpp"
#include "aeongui/FontDatabase.hpp"
#include "aeongui/RasterImage.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/dom/Document.hpp"

// ==== AeonGUI Initialize / Finalize ====

TEST ( AeonGUITest, InitializeDoesNotThrow )
{
    ASSERT_NO_THROW ( AeonGUI::Initialize ( 0, nullptr ) );
    AeonGUI::Finalize();
}

TEST ( AeonGUITest, FinalizeAfterInitializeDoesNotThrow )
{
    AeonGUI::Initialize ( 0, nullptr );
    ASSERT_NO_THROW ( AeonGUI::Finalize() );
}

TEST ( AeonGUITest, FinalizeWithoutInitializeDoesNotThrow )
{
    ASSERT_NO_THROW ( AeonGUI::Finalize() );
}

// ==== FontDatabase error paths ====

class FontDatabaseErrorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        AeonGUI::FontDatabase::Finalize();
    }
    void TearDown() override
    {
        AeonGUI::FontDatabase::Finalize();
    }
};

TEST_F ( FontDatabaseErrorTest, CreateContextBeforeInitializeThrows )
{
    EXPECT_THROW ( AeonGUI::FontDatabase::CreateContext(), std::runtime_error );
}

TEST_F ( FontDatabaseErrorTest, AddFontFileBeforeInitializeThrows )
{
    EXPECT_THROW ( AeonGUI::FontDatabase::AddFontFile ( "/nonexistent/font.ttf" ), std::runtime_error );
}

TEST_F ( FontDatabaseErrorTest, AddFontDirectoryBeforeInitializeThrows )
{
    EXPECT_THROW ( AeonGUI::FontDatabase::AddFontDirectory ( "/nonexistent/dir" ), std::runtime_error );
}

TEST_F ( FontDatabaseErrorTest, InitializeTwiceIsIdempotent )
{
    ASSERT_NO_THROW ( AeonGUI::FontDatabase::Initialize() );
    ASSERT_NO_THROW ( AeonGUI::FontDatabase::Initialize() );
}

TEST_F ( FontDatabaseErrorTest, FinalizeTwiceIsIdempotent )
{
    AeonGUI::FontDatabase::Initialize();
    ASSERT_NO_THROW ( AeonGUI::FontDatabase::Finalize() );
    ASSERT_NO_THROW ( AeonGUI::FontDatabase::Finalize() );
}

TEST_F ( FontDatabaseErrorTest, AddFontFileWithInvalidPathThrows )
{
    AeonGUI::FontDatabase::Initialize();
    EXPECT_THROW ( AeonGUI::FontDatabase::AddFontFile ( "/nonexistent/font.ttf" ), std::runtime_error );
}

// ==== RasterImage error paths ====

TEST ( RasterImageErrorTest, LoadFromMemoryNullDataThrows )
{
    AeonGUI::RasterImage image;
    EXPECT_THROW ( image.LoadFromMemory ( nullptr, 100 ), std::runtime_error );
}

TEST ( RasterImageErrorTest, LoadFromMemoryZeroSizeThrows )
{
    AeonGUI::RasterImage image;
    const uint8_t data[4] = {0};
    EXPECT_THROW ( image.LoadFromMemory ( data, 0 ), std::runtime_error );
}

TEST ( RasterImageErrorTest, LoadFromMemoryTooSmallThrows )
{
    AeonGUI::RasterImage image;
    const uint8_t data[2] = {0x00, 0x01};
    EXPECT_THROW ( image.LoadFromMemory ( data, 2 ), std::runtime_error );
}

TEST ( RasterImageErrorTest, LoadFromMemoryUnknownFormatThrows )
{
    AeonGUI::RasterImage image;
    std::vector<uint8_t> data ( 128, 0x00 );
    EXPECT_THROW ( image.LoadFromMemory ( data.data(), data.size() ), std::runtime_error );
}

TEST ( RasterImageErrorTest, LoadFromFileMissingFileThrows )
{
    AeonGUI::RasterImage image;
    EXPECT_THROW ( image.LoadFromFile ( "/nonexistent/path.pcx" ), std::runtime_error );
}

TEST ( RasterImageErrorTest, LoadFromFileEmptyFileThrows )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-empty.pcx";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
    }
    AeonGUI::RasterImage image;
    EXPECT_THROW ( image.LoadFromFile ( path.string() ), std::runtime_error );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

// ==== Color error paths ====

TEST ( ColorErrorTest, ValidNamedColorDoesNotThrow )
{
    ASSERT_NO_THROW ( AeonGUI::Color ( "red" ) );
    AeonGUI::Color c ( "red" );
    EXPECT_DOUBLE_EQ ( c.R(), 1.0 );
    EXPECT_DOUBLE_EQ ( c.G(), 0.0 );
    EXPECT_DOUBLE_EQ ( c.B(), 0.0 );
}

TEST ( ColorErrorTest, ValidHex6ColorDoesNotThrow )
{
    ASSERT_NO_THROW ( AeonGUI::Color ( "#00ff00" ) );
    AeonGUI::Color c ( "#00ff00" );
    EXPECT_DOUBLE_EQ ( c.G(), 1.0 );
}

TEST ( ColorErrorTest, InvalidColorStringThrows )
{
    EXPECT_THROW ( AeonGUI::Color ( "notacolor" ), std::runtime_error );
}

TEST ( ColorErrorTest, EmptyStringThrows )
{
    EXPECT_THROW ( AeonGUI::Color ( std::string{} ), std::runtime_error );
}

// ==== Document error paths ====

TEST ( DocumentErrorTest, LoadNonExistentFileThrows )
{
    AeonGUI::DOM::Document document;
    EXPECT_THROW ( document.Load ( "/nonexistent/file.svg" ), std::runtime_error );
}

TEST ( DocumentErrorTest, LoadValidSvgDoesNotThrow )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-valid.svg";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
        f << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
          << R"(<rect x="0" y="0" width="50" height="50"/>)"
          << R"(</svg>)";
    }
    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

// ==== SVGScriptElement error paths ====

TEST ( SVGScriptElementErrorTest, NativeScriptMissingHrefThrows )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-script-nohref.svg";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
        f << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
          << R"(<script type="native"/>)"
          << R"(</svg>)";
    }
    AeonGUI::DOM::Document document;
    EXPECT_THROW ( document.Load ( path.string() ), std::runtime_error );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

TEST ( SVGScriptElementErrorTest, NonNativeTypeIgnored )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-script-js.svg";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
        f << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
          << R"(<script type="text/javascript" href="foo.js"/>)"
          << R"(</svg>)";
    }
    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

TEST ( SVGScriptElementErrorTest, MissingTypeAttributeIgnored )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-script-notype.svg";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
        f << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
          << R"(<script href="foo"/>)"
          << R"(</svg>)";
    }
    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

TEST ( SVGScriptElementErrorTest, NonexistentPluginDoesNotThrow )
{
    auto path = std::filesystem::temp_directory_path() / "aeongui-script-missing.svg";
    {
        std::ofstream f ( path, std::ios::binary | std::ios::out );
        f << R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
          << R"(<script type="native" href="absolutely_nonexistent_lib"/>)"
<< R"(</svg>)";
    }
    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );
    std::error_code ec;
    std::filesystem::remove ( path, ec );
}
