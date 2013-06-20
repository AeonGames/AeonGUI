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
    /*! \brief Raster Font class.
        \note This class has to change to accomodate for multiple font formats.
    */
    class Font
    {
    public:
        /// Font Glyph data.
        struct Glyph
        {
            uint32_t charcode; ///< Wide character character code.
            uint16_t width;    ///< Glyph image width.
            uint16_t height;   ///< Glyph image height.
            uint16_t min[2];   ///< Bounding box minimums.
            uint16_t max[2];   ///< Bounding box maximums.
            float normalized_min[2]; ///< Bounding box normalized minimums.
            float normalized_max[2]; ///< Bounding box normalized maximums.
            int16_t top;             ///< Glyph top.
            int16_t left;            ///< Glyph left.
            int16_t advance[2];      ///< Glyph x and y advance.
        };

        Font();

        /*! \brief Load font from memory buffer.
            \param data Font file memory buffer.
            \param size Font file memory buffer size in bytes.
            \return true if load was succesful, false otherwise.
            */
        bool Load ( void* data, size_t size );

        /*! \brief Load font from file.
            \param filename font file file path.
            \return true if load was succesful, false otherwise.
            */
        bool Load ( const char* filename );

        virtual ~Font();

        /*! \brief Get number of glyphs in font.
            \return Glyph count.
        */
        uint32_t GetGlyphCount();

        /*! \brief Get font bitmap width.
            \return Font bitmap width.*/
        uint32_t GetMapWidth();

        /*! \brief Get font bitmap height.
            \return Font bitmap height.*/
        uint32_t GetMapHeight();

        /*! \brief Get font nominal width.
            \return Font bitmap width.*/
        uint16_t GetNominalWidth();

        /*! \brief Get font nominal height.
            \return Font bitmap height.*/
        uint16_t GetNominalHeight();

        /*! \brief Get font ascender value.
            \return Font ascender value.*/
        int16_t GetAscender();

        /*! \brief Get font descender value.
            \return Font descender value.*/
        int16_t GetDescender() const;

        /*! \brief Get font height or size.
            \return Font height.*/
        uint16_t GetHeight() const;

        /*! \brief Get font maximum advance.
            \return Font maximum advance.*/
        int16_t GetMaxAdvance();

        /*! \brief Get a specific glyph.
            \param charcode Unicode character code.
            \return pointer to glyph structure or NULL if not found.*/
        Glyph* GetGlyph ( wchar_t charcode );

        /*! \brief Get glyph bitmap.
            \return pointer to glyph bitmap buffer.*/
        const uint8_t* GetGlyphMap();
    protected:
        uint32_t glyphcount;    ///< Number of glyphs contained in the font.
        uint32_t map_width;     ///< Font bitmap width.
        uint32_t map_height;    ///< Font bitmap height.
        uint16_t nominal_width; ///< Font nominal width.
        uint16_t nominal_height;///< Font nominal height.
        int16_t ascender;       ///< Font ascender.
        int16_t descender;      ///< Font descender.
        uint16_t height;        ///< Font height or size
        int16_t max_advance;    ///< Font maximum advance.

        Glyph* glyphdata;       ///< Pointer to glyph data array.
        uint8_t* glyphmap;      ///< The font bitmap buffer.
    };
}
#endif
