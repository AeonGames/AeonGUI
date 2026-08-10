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
#include <algorithm>
#include <cstdlib>
#include "aeongui/dom/HTMLTextAreaElement.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/Canvas.hpp"
#include "PangoTextLayout.hpp"
#include <pango/pango.h>

namespace AeonGUI
{
    namespace DOM
    {
        namespace
        {
            bool IsContinuationByte ( char aByte )
            {
                return ( static_cast<unsigned char> ( aByte ) & 0xC0u ) == 0x80u;
            }

            size_t PreviousCodePoint ( const DOMString& aText, size_t aIndex )
            {
                if ( aIndex == 0 )
                {
                    return 0;
                }
                size_t index = aIndex - 1;
                while ( index > 0 && IsContinuationByte ( aText[index] ) )
                {
                    --index;
                }
                return index;
            }

            size_t NextCodePoint ( const DOMString& aText, size_t aIndex )
            {
                if ( aIndex >= aText.size() )
                {
                    return aText.size();
                }
                size_t index = aIndex + 1;
                while ( index < aText.size() && IsContinuationByte ( aText[index] ) )
                {
                    ++index;
                }
                return index;
            }

            size_t CountCodePoints ( const DOMString& aText )
            {
                size_t count = 0;
                for ( char c : aText )
                {
                    if ( !IsContinuationByte ( c ) )
                    {
                        ++count;
                    }
                }
                return count;
            }

            bool IsPrintableKey ( const DOMString& aKey )
            {
                if ( aKey.empty() || CountCodePoints ( aKey ) != 1 )
                {
                    return false;
                }
                return aKey.size() > 1 || static_cast<unsigned char> ( aKey[0] ) >= 0x20u;
            }

            uint32_t ParsePositiveAttribute ( const DOMString* aValue, uint32_t aFallback )
            {
                if ( !aValue )
                {
                    return aFallback;
                }
                const long parsed = std::strtol ( aValue->c_str(), nullptr, 10 );
                return parsed > 0 ? static_cast<uint32_t> ( parsed ) : aFallback;
            }
        }

        HTMLTextAreaElement::HTMLTextAreaElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLFormControlElement { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLTextAreaElement::~HTMLTextAreaElement() = default;

        uint32_t HTMLTextAreaElement::rows() const
        {
            return ParsePositiveAttribute ( getAttribute ( "rows" ), 2u );
        }

        uint32_t HTMLTextAreaElement::cols() const
        {
            return ParsePositiveAttribute ( getAttribute ( "cols" ), 20u );
        }

        DOMString HTMLTextAreaElement::placeholder() const
        {
            const DOMString* attribute = getAttribute ( "placeholder" );
            return attribute ? *attribute : DOMString{};
        }

        bool HTMLTextAreaElement::readOnly() const
        {
            return getAttribute ( "readonly" ) != nullptr;
        }

        DOMString HTMLTextAreaElement::defaultValue() const
        {
            return textContent();
        }

        size_t HTMLTextAreaElement::caretPosition() const
        {
            return mCaret;
        }

        DOMString HTMLTextAreaElement::value() const
        {
            return mValueInitialized ? mValue : defaultValue();
        }

        void HTMLTextAreaElement::setValue ( const DOMString& aValue )
        {
            mValueInitialized = true;
            if ( mValue == aValue )
            {
                return;
            }
            mValue = aValue;
            mCaret = std::min ( mCaret, mValue.size() );
            RequestRedraw();
        }

        void HTMLTextAreaElement::Reset()
        {
            mValueInitialized = false;
            mValue.clear();
            mCaret = 0;
            RequestRedraw();
        }

        bool HTMLTextAreaElement::IsTextEditable() const
        {
            return !readOnly() && !isDisabled();
        }

        bool HTMLTextAreaElement::HandleKey ( const DOMString& aKey )
        {
            if ( isDisabled() )
            {
                return false;
            }
            EnsureValueInitialized();
            if ( aKey == "ArrowLeft" )
            {
                mCaret = PreviousCodePoint ( mValue, mCaret );
                RequestRedraw();
                return true;
            }
            if ( aKey == "ArrowRight" )
            {
                mCaret = NextCodePoint ( mValue, mCaret );
                RequestRedraw();
                return true;
            }
            if ( aKey == "Home" )
            {
                mCaret = mValue.rfind ( '\n', mCaret > 0 ? mCaret - 1 : 0 );
                mCaret = ( mCaret == DOMString::npos ) ? 0 : mCaret + 1;
                RequestRedraw();
                return true;
            }
            if ( aKey == "End" )
            {
                const size_t line_end = mValue.find ( '\n', mCaret );
                mCaret = ( line_end == DOMString::npos ) ? mValue.size() : line_end;
                RequestRedraw();
                return true;
            }
            if ( readOnly() )
            {
                return false;
            }
            if ( aKey == "Backspace" )
            {
                if ( mCaret == 0 )
                {
                    return true;
                }
                const size_t start = PreviousCodePoint ( mValue, mCaret );
                mValue.erase ( start, mCaret - start );
                mCaret = start;
            }
            else if ( aKey == "Delete" )
            {
                if ( mCaret >= mValue.size() )
                {
                    return true;
                }
                const size_t end = NextCodePoint ( mValue, mCaret );
                mValue.erase ( mCaret, end - mCaret );
            }
            else if ( aKey == "Enter" )
            {
                mValue.insert ( mCaret, "\n" );
                ++mCaret;
            }
            else if ( IsPrintableKey ( aKey ) )
            {
                mValue.insert ( mCaret, aKey );
                mCaret += aKey.size();
            }
            else
            {
                return false;
            }
            RequestRedraw();
            Event inputEvent ( "input", EventInit{true, false, false} );
            dispatchEvent ( inputEvent );
            return true;
        }

        bool HTMLTextAreaElement::GetIntrinsicContentSize ( float& aWidth, float& aHeight ) const
        {
            const FontSpec font = GetFontSpec();
            PangoTextLayout layout;
            layout.SetFontFamily ( font.family );
            layout.SetFontSize   ( font.size );
            layout.SetFontWeight ( font.weight );
            layout.SetFontStyle  ( font.style );
            layout.SetText ( "0" );
            aWidth  = static_cast<float> ( layout.GetTextWidth() * cols() );
            aHeight = static_cast<float> ( layout.GetTextHeight() * rows() );
            return true;
        }

        void HTMLTextAreaElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            HTMLFormControlElement::onAttributeChanged ( aName, aValue );
            if ( aName == "disabled" || aName == "readonly" )
            {
                ReselectCSS();
            }
        }

        void HTMLTextAreaElement::EnsureValueInitialized()
        {
            if ( mValueInitialized )
            {
                return;
            }
            mValue = defaultValue();
            mCaret = mValue.size();
            mValueInitialized = true;
        }

        void HTMLTextAreaElement::DrawStart ( Canvas& aCanvas ) const
        {
            // Skip HTMLElement::DrawStart's inline text pass: the text
            // children are the *default value*, not rendered content.
            PaintBox ( aCanvas );

            const LayoutBox& box = GetLayoutBox();
            if ( box.contentWidth <= 0.0f || box.contentHeight <= 0.0f )
            {
                return;
            }

            const DOMString text = value();
            const bool showing_placeholder = text.empty();
            const DOMString painted = showing_placeholder ? placeholder() : text;

            const FontSpec font = GetFontSpec();
            PangoTextLayout layout;
            layout.SetFontFamily ( font.family );
            layout.SetFontSize   ( font.size );
            layout.SetFontWeight ( font.weight );
            layout.SetFontStyle  ( font.style );
            layout.SetWrapWidth  ( box.contentWidth );
            layout.SetText ( painted.empty() ? DOMString{"0"} : painted );

            const ColorAttr previous_fill = aCanvas.GetFillColor();
            aCanvas.Save();
            aCanvas.SetClipRect ( box.contentX, box.contentY,
                                  box.contentWidth, box.contentHeight );

            ColorAttr text_color{};
            if ( !painted.empty() && ResolveTextColor ( text_color ) )
            {
                if ( showing_placeholder )
                {
                    text_color = ColorAttr{ Color{ 0xFF767676u } };
                }
                aCanvas.SetFillColor ( text_color );
                PangoLayout* pango_layout = layout.GetPangoLayout();
                PangoLayoutIter* iter = pango_layout_get_iter ( pango_layout );
                do
                {
                    PangoLayoutLine* line = pango_layout_iter_get_line_readonly ( iter );
                    if ( !line || line->length <= 0 )
                    {
                        continue;
                    }
                    const double baseline =
                        box.contentY +
                        static_cast<double> ( pango_layout_iter_get_baseline ( iter ) ) / PANGO_SCALE;
                    aCanvas.DrawText (
                        painted.substr ( static_cast<size_t> ( line->start_index ),
                                         static_cast<size_t> ( line->length ) ),
                        box.contentX, baseline,
                        font.family, font.size, font.weight, font.style );
                }
                while ( pango_layout_iter_next_line ( iter ) );
                pango_layout_iter_free ( iter );
            }

            if ( isFocus() && !isDisabled() && !readOnly() )
            {
                PangoRectangle caret{};
                pango_layout_index_to_pos (
                    layout.GetPangoLayout(),
                    showing_placeholder ? 0 : static_cast<int> ( mCaret ),
                    &caret );
                PaintCaret ( aCanvas,
                             box.contentX + static_cast<double> ( caret.x ) / PANGO_SCALE,
                             box.contentY + static_cast<double> ( caret.y ) / PANGO_SCALE,
                             static_cast<double> ( caret.height ) / PANGO_SCALE );
            }

            aCanvas.Restore();
            aCanvas.SetFillColor ( previous_fill );
        }
    }
}
