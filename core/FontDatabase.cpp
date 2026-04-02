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

#include "aeongui/FontDatabase.hpp"
#include <fontconfig/fontconfig.h>
#include <pango/pango.h>
#include <pango/pangofc-fontmap.h>
#ifdef AEONGUI_USE_SKIA
#include <pango/pangoft2.h>
#else
#include <pango/pangocairo.h>
#include <cairo.h>
#endif
#include <iostream>
#include <filesystem>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#include <climits>
#endif

namespace AeonGUI
{
    std::recursive_mutex FontDatabase::sMutex;
    FcConfig* FontDatabase::sFcConfig = nullptr;
    PangoFontMap* FontDatabase::sFontMap = nullptr;

    static std::filesystem::path GetExecutableDir()
    {
#ifdef _WIN32
        wchar_t buf[MAX_PATH] {};
        GetModuleFileNameW ( nullptr, buf, MAX_PATH );
        return std::filesystem::path{buf}.parent_path();
#elif defined(__APPLE__)
        char buf[PATH_MAX] {};
        uint32_t size = sizeof ( buf );
        if ( _NSGetExecutablePath ( buf, &size ) == 0 )
        {
            return std::filesystem::canonical ( buf ).parent_path();
        }
        return {};
#else
        char buf[PATH_MAX] {};
        ssize_t len = readlink ( "/proc/self/exe", buf, sizeof ( buf ) - 1 );
        if ( len > 0 )
        {
            buf[len] = '\0';
            return std::filesystem::path{buf}.parent_path();
        }
        return {};
#endif
    }

    static void LogAvailableFonts ( FcConfig* aConfig )
    {
        FcPattern* pattern = FcPatternCreate();
        FcObjectSet* objectSet = FcObjectSetBuild ( FC_FAMILY, FC_STYLE, FC_FILE, nullptr );
        FcFontSet* fontSet = FcFontList ( aConfig, pattern, objectSet );
        if ( fontSet != nullptr )
        {
            std::cout << "FontDatabase: " << fontSet->nfont << " font(s) available:" << std::endl;
            for ( int i = 0; i < fontSet->nfont; ++i )
            {
                FcChar8* family = nullptr;
                FcChar8* style = nullptr;
                FcChar8* file = nullptr;
                FcPatternGetString ( fontSet->fonts[i], FC_FAMILY, 0, &family );
                FcPatternGetString ( fontSet->fonts[i], FC_STYLE, 0, &style );
                FcPatternGetString ( fontSet->fonts[i], FC_FILE,   0, &file );
                std::cout << "  [" << i << "] "
                          << ( family ? reinterpret_cast<const char*> ( family ) : "?" ) << " "
                          << ( style  ? reinterpret_cast<const char*> ( style )  : "" )  << "  ("
                          << ( file   ? reinterpret_cast<const char*> ( file )   : "?" ) << ")"
                          << std::endl;
            }
            FcFontSetDestroy ( fontSet );
        }
        else
        {
            std::cout << "FontDatabase: 0 font(s) available." << std::endl;
        }
        if ( objectSet != nullptr )
        {
            FcObjectSetDestroy ( objectSet );
        }
        FcPatternDestroy ( pattern );
    }

    bool FontDatabase::Initialize()
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        if ( sFcConfig != nullptr )
        {
            return true;
        }

        sFcConfig = FcConfigCreate();
        if ( sFcConfig == nullptr )
        {
            std::cerr << "FontDatabase: Failed to create FcConfig" << std::endl;
            return false;
        }

        // Do NOT load system configuration — we want a local-only database.
        // FcConfigSetCurrent is NOT called so the system default remains untouched for
        // other libraries; we only use sFcConfig explicitly via our font map.

#ifdef AEONGUI_USE_SKIA
        sFontMap = pango_ft2_font_map_new();
#else
        sFontMap = pango_cairo_font_map_new_for_font_type ( CAIRO_FONT_TYPE_FT );
#endif
        if ( sFontMap == nullptr )
        {
            FcConfigDestroy ( sFcConfig );
            sFcConfig = nullptr;
            std::cerr << "FontDatabase: Failed to create PangoFcFontMap" << std::endl;
            return false;
        }

        pango_fc_font_map_set_config ( PANGO_FC_FONT_MAP ( sFontMap ), sFcConfig );

        // Auto-scan for a fonts/ directory next to the executable.
        std::filesystem::path fontsDir = GetExecutableDir() / "fonts";
        if ( std::filesystem::is_directory ( fontsDir ) )
        {
            std::cout << "FontDatabase: Auto-loading fonts from " << fontsDir.string() << std::endl;
            AddFontDirectory ( fontsDir.string() );
        }
        else
        {
            std::cout << "FontDatabase: No fonts/ directory found at " << fontsDir.string() << std::endl;
        }

        LogAvailableFonts ( sFcConfig );
        return true;
    }

    void FontDatabase::Finalize()
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        if ( sFontMap != nullptr )
        {
            g_object_unref ( sFontMap );
            sFontMap = nullptr;
        }
        if ( sFcConfig != nullptr )
        {
            FcConfigDestroy ( sFcConfig );
            sFcConfig = nullptr;
        }
    }

    bool FontDatabase::AddFontDirectory ( const std::string& aPath )
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        if ( sFcConfig == nullptr )
        {
            std::cerr << "FontDatabase: Not initialized" << std::endl;
            return false;
        }
        if ( !FcConfigAppFontAddDir ( sFcConfig, reinterpret_cast<const FcChar8*> ( aPath.c_str() ) ) )
        {
            std::cerr << "FontDatabase: Failed to add font directory: " << aPath << std::endl;
            return false;
        }
        // Notify pango that the font configuration changed.
        pango_fc_font_map_config_changed ( PANGO_FC_FONT_MAP ( sFontMap ) );
        return true;
    }

    bool FontDatabase::AddFontFile ( const std::string& aPath )
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        if ( sFcConfig == nullptr )
        {
            std::cerr << "FontDatabase: Not initialized" << std::endl;
            return false;
        }
        if ( !FcConfigAppFontAddFile ( sFcConfig, reinterpret_cast<const FcChar8*> ( aPath.c_str() ) ) )
        {
            std::cerr << "FontDatabase: Failed to add font file: " << aPath << std::endl;
            return false;
        }
        pango_fc_font_map_config_changed ( PANGO_FC_FONT_MAP ( sFontMap ) );
        return true;
    }

    FcConfig* FontDatabase::GetFcConfig()
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        return sFcConfig;
    }

    PangoFontMap* FontDatabase::GetFontMap()
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        return sFontMap;
    }

    PangoContext* FontDatabase::CreateContext()
    {
        std::lock_guard<std::recursive_mutex> lock ( sMutex );
        if ( sFontMap == nullptr )
        {
            return nullptr;
        }
        return pango_font_map_create_context ( sFontMap );
    }
}
