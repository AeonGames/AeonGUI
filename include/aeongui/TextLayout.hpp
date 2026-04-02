/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_TEXTLAYOUT_H
#define AEONGUI_TEXTLAYOUT_H
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "aeongui/Platform.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/Attribute.hpp"
#include "aeongui/Matrix2x3.hpp"
namespace AeonGUI
{
    /** @brief Abstract text layout interface.
     *
     *  Abstracts PangoLayout to allow multiple text‐layout back-ends
     *  to be used interchangeably.
     */
    class TextLayout
    {
    public:
        /** @brief Virtual destructor. */
        DLL virtual ~TextLayout() = 0;
        /** @brief Set the text content to lay out.
         *  @param aText UTF-8 text string.
         */
        virtual void SetText ( const std::string& aText ) = 0;
        /** @brief Set the font family.
         *  @param aFamily Font family name (e.g. "sans-serif").
         */
        virtual void SetFontFamily ( const std::string& aFamily ) = 0;
        /** @brief Set the font size.
         *  @param aSize Font size in CSS pixels.
         */
        virtual void SetFontSize ( double aSize ) = 0;
        /** @brief Set the font weight.
         *  @param aWeight CSS font weight (400 = normal, 700 = bold).
         */
        virtual void SetFontWeight ( int aWeight ) = 0;
        /** @brief Set the font style.
         *  @param aStyle 0 = normal, 1 = italic, 2 = oblique.
         */
        virtual void SetFontStyle ( int aStyle ) = 0;
        /** @brief Get the logical width of the laid-out text.
         *  @return Width in CSS pixels.
         */
        virtual double GetTextWidth() const = 0;
        /** @brief Get the logical height of the laid-out text.
         *  @return Height in CSS pixels.
         */
        virtual double GetTextHeight() const = 0;
        /** @brief Get the baseline offset from the top.
         *  @return Baseline in CSS pixels.
         */
        virtual double GetBaseline() const = 0;
        /** Get the x-offset of the character at the given index.
         *  @param aIndex UTF-8 byte index.
         *  @return x offset in CSS pixels.
         */
        virtual double GetCharOffsetX ( long aIndex ) const = 0;
    };
}
#endif