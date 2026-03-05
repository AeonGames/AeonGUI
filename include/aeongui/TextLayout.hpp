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
    /** Text layout interface
     * This abstracts PangoLayout the same way Canvas abstracts Cairo surface in order to allow for multiple backends to be used interchangeably in the future.
    */
    class TextLayout
    {
    public:
        DLL virtual ~TextLayout() = 0;
        virtual void SetText ( const std::string& aText ) = 0;
        virtual void SetFontFamily ( const std::string& aFamily ) = 0;
        virtual void SetFontSize ( double aSize ) = 0;
        virtual void SetFontWeight ( int aWeight ) = 0;
        virtual void SetFontStyle ( int aStyle ) = 0;
        virtual double GetTextWidth() const = 0;
        virtual double GetTextHeight() const = 0;
        virtual double GetBaseline() const = 0;
        /** Get the x-offset of the character at the given index.
         *  @param aIndex UTF-8 byte index.
         *  @return x offset in CSS pixels.
         */
        virtual double GetCharOffsetX ( long aIndex ) const = 0;
    };
}
#endif