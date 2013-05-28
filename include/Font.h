#ifndef AEONGUI_FONT_H
#define AEONGUI_FONT_H
/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include <cstddef>
#include <cassert>
#include "Integer.h"
namespace AeonGUI
{
    class Font
    {
    public:
        struct Glyph
        {
            uint32_t charcode;
            uint16_t width;
            uint16_t height;
            uint16_t min[2];
            uint16_t max[2];
            float normalized_min[2];
            float normalized_max[2];
            int16_t top;
            int16_t left;
            int16_t advance[2];
        };
        Font ( void* data, size_t size );
        Font ( const char* filename );
        virtual ~Font()
        {
            if ( NULL != glyphdata )
            {
                delete[] glyphdata;
            }
            if ( NULL != glyphmap )
            {
                delete[] glyphmap;
            }
        }
        inline uint32_t GetGlyphCount()
        {
            return glyphcount;
        }
        inline uint32_t GetMapWidth()
        {
            return map_width;
        }
        inline uint32_t GetMapHeight()
        {
            return map_height;
        }
        inline uint16_t GetNominalWidth()
        {
            return nominal_width;
        }
        inline uint16_t GetNominalHeight()
        {
            return nominal_height;
        }
        inline int16_t GetAscender()
        {
            return ascender;
        }
        inline int16_t GetDescender() const
        {
            return descender;
        }
        inline uint16_t GetHeight() const
        {
            return height;
        }
        inline int16_t GetMaxAdvance()
        {
            return max_advance;
        }
        Glyph* GetGlyph ( wchar_t charcode );
        inline const uint8_t* GetGlyphMap()
        {
            return glyphmap;
        }
    protected:
        bool isgood;
        uint32_t glyphcount;
        uint32_t map_width;
        uint32_t map_height;
        uint16_t nominal_width;
        uint16_t nominal_height;
        int16_t ascender;
        int16_t descender;
        uint16_t height;
        int16_t max_advance;

        Glyph* glyphdata;
        uint8_t* glyphmap;
    };
}
#endif
