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
#ifndef AEONGUI_FONTDATABASE_H
#define AEONGUI_FONTDATABASE_H

#include <string>
#include <vector>
#include <mutex>
#include "aeongui/Platform.hpp"

struct _PangoFontMap;
typedef struct _PangoFontMap PangoFontMap;
struct _PangoContext;
typedef struct _PangoContext PangoContext;
struct _FcConfig;
typedef struct _FcConfig FcConfig;

namespace AeonGUI
{
    /**
     * FontDatabase provides a local font database for AeonGUI,
     * independent of system-installed fonts.
     * It uses FcConfig to load fonts from application-provided directories.
     * It also manages a PangoFontMap backed by the local FcConfig.
     *
     * This is a singleton; call Initialize()/Finalize() once.
     */
    class FontDatabase
    {
    public:
        /** @brief Initialize the font database. Call once at startup.
         *  @throws std::runtime_error on failure.
         */
        DLL static void Initialize();
        /// Finalize the font database. Call once at shutdown.
        DLL static void Finalize();
        /** @brief Add a directory of font files (.ttf, .otf, etc.) to the database.
         *  @param aPath Path to the font directory.
         *  @throws std::runtime_error on failure.
         */
        DLL static void AddFontDirectory ( const std::string& aPath );
        /** @brief Add a single font file to the database.
         *  @param aPath Path to the font file.
         *  @throws std::runtime_error on failure.
         */
        DLL static void AddFontFile ( const std::string& aPath );
        /** @brief Get the FcConfig used by the font database.
         *  @return Pointer to the FcConfig.
         */
        DLL static FcConfig* GetFcConfig();
        /** @brief Get the PangoFontMap backed by the local font database.
         *  @return Pointer to the PangoFontMap.
         */
        DLL static PangoFontMap* GetFontMap();
        /** @brief Create a new PangoContext from the local font map.
         *  @return Pointer to a new PangoContext.
         */
        DLL static PangoContext* CreateContext();
    private:
        FontDatabase() = delete;
        static std::recursive_mutex& GetMutex();
        static FcConfig* sFcConfig;
        static PangoFontMap* sFontMap;
    };
}
#endif
