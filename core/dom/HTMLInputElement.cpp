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
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include "aeongui/dom/HTMLInputElement.hpp"
#include "aeongui/dom/HTMLFormElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include "AsciiCase.hpp"
#include "PangoTextLayout.hpp"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        namespace
        {
            /// Bullet (U+2022) used to mask password field contents.
            constexpr const char kPasswordBullet[] = "\xE2\x80\xA2";

            /// Length of the longest recognized `type` keyword
            /// ("password" / "checkbox").  Anything longer cannot match
            /// and is rejected without touching the lookup table.
            constexpr size_t kMaxTypeKeyword = 8;

            /// ASCII-lowercase @p aValue into @p aBuffer so the keyword
            /// lookup needs no heap allocation.  Returns an empty view
            /// when the input is too long to be a keyword.
            std::string_view LowercaseKeyword ( const DOMString& aValue,
                                                std::array<char, kMaxTypeKeyword>& aBuffer )
            {
                if ( aValue.empty() || aValue.size() > aBuffer.size() )
                {
                    return {};
                }
                for ( size_t i = 0; i < aValue.size(); ++i )
                {
                    aBuffer[i] = static_cast<char> (
                                     std::tolower ( static_cast<unsigned char> ( aValue[i] ) ) );
                }
                return std::string_view{ aBuffer.data(), aValue.size() };
            }

            const std::unordered_map<std::string_view, HTMLInputElement::Type>& TypeKeywords()
            {
                using Type = HTMLInputElement::Type;
                static const std::unordered_map<std::string_view, Type> keywords
                {
                    { "text",     Type::Text     },
                    { "password", Type::Password },
                    { "search",   Type::Search   },
                    { "tel",      Type::Tel      },
                    { "url",      Type::Url      },
                    { "email",    Type::Email    },
                    { "number",   Type::Number   },
                    { "range",    Type::Range    },
                    { "hidden",   Type::Hidden   },
                    { "checkbox", Type::Checkbox },
                    { "radio",    Type::Radio    },
                    { "button",   Type::Button   },
                    { "submit",   Type::Submit   },
                    { "reset",    Type::Reset    },
                    // `file` and `image` need filesystem access and image
                    // submission coordinates respectively; both are out
                    // of scope, so they parse but never render.
                    { "file",     Type::Unsupported },
                    { "image",    Type::Unsupported },
                };
                return keywords;
            }

            HTMLInputElement::Type ParseType ( const DOMString& aValue )
            {
                std::array<char, kMaxTypeKeyword> buffer{};
                const auto& keywords = TypeKeywords();
                const auto found = keywords.find ( LowercaseKeyword ( aValue, buffer ) );
                // Types the browser does not know fall back to text.
                return found != keywords.end() ? found->second : HTMLInputElement::Type::Text;
            }

            bool IsTextType ( HTMLInputElement::Type aType )
            {
                using Type = HTMLInputElement::Type;
                switch ( aType )
                {
                case Type::Text:
                case Type::Password:
                case Type::Search:
                case Type::Tel:
                case Type::Url:
                case Type::Email:
                case Type::Number:
                    return true;
                default:
                    return false;
                }
            }

            bool IsButtonType ( HTMLInputElement::Type aType )
            {
                using Type = HTMLInputElement::Type;
                return aType == Type::Button || aType == Type::Submit || aType == Type::Reset;
            }

            /// UTF-8 continuation bytes have the 10xxxxxx bit pattern;
            /// stepping over them keeps the caret on code point limits.
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

            /// A DOM `key` value that carries exactly one code point is
            /// a character to insert; anything longer ("Enter",
            /// "Backspace", "Shift", ...) is a named key.
            bool IsPrintableKey ( const DOMString& aKey )
            {
                if ( aKey.empty() || CountCodePoints ( aKey ) != 1 )
                {
                    return false;
                }
                // Reject the C0 controls that arrive as single characters.
                return aKey.size() > 1 || static_cast<unsigned char> ( aKey[0] ) >= 0x20u;
            }

            css_computed_style* MutableStyleOf ( const HTMLElement& aElement )
            {
                css_select_results* results = aElement.GetComputedStyles();
                return results ? results->styles[CSS_PSEUDO_ELEMENT_NONE] : nullptr;
            }

            /// Parse a floating point attribute, returning @p aFallback
            /// when the attribute is missing or not a valid number.
            double ParseNumber ( const DOMString* aValue, double aFallback )
            {
                if ( !aValue || aValue->empty() )
                {
                    return aFallback;
                }
                char* end{nullptr};
                const double parsed = std::strtod ( aValue->c_str(), &end );
                if ( end == aValue->c_str() || !std::isfinite ( parsed ) )
                {
                    return aFallback;
                }
                return parsed;
            }

            /// Shortest round-trippable-enough decimal form, so a slider
            /// at 0.5 submits "0.5" rather than "0.500000".
            DOMString NumberToString ( double aValue )
            {
                DOMString text = std::to_string ( aValue );
                const size_t dot = text.find ( '.' );
                if ( dot == DOMString::npos )
                {
                    return text;
                }
                text.erase ( text.find_last_not_of ( '0' ) + 1 );
                if ( !text.empty() && text.back() == '.' )
                {
                    text.pop_back();
                }
                return text;
            }
        }

        HTMLInputElement::HTMLInputElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLFormControlElement { aTagName, std::move ( aAttributes ), aParent }
        {
            const DOMString* type_attribute = getAttribute ( "type" );
            mType = type_attribute ? ParseType ( *type_attribute ) : Type::Text;
            mValue = defaultValue();
            mCaret = mValue.size();
            mChecked = defaultChecked();
        }

        HTMLInputElement::~HTMLInputElement() = default;

        HTMLInputElement::Type HTMLInputElement::type() const
        {
            return mType;
        }

        DOMString HTMLInputElement::defaultValue() const
        {
            const DOMString* attribute = getAttribute ( "value" );
            return attribute ? *attribute : DOMString{};
        }

        bool HTMLInputElement::defaultChecked() const
        {
            return getAttribute ( "checked" ) != nullptr;
        }

        bool HTMLInputElement::checked() const
        {
            return mChecked;
        }

        void HTMLInputElement::setChecked ( bool aChecked )
        {
            if ( mChecked == aChecked )
            {
                return;
            }
            mChecked = aChecked;
            if ( mChecked && mType == Type::Radio )
            {
                ClearRadioGroup();
            }
            ReselectCSS();
            RequestRedraw();
        }

        DOMString HTMLInputElement::placeholder() const
        {
            const DOMString* attribute = getAttribute ( "placeholder" );
            return attribute ? *attribute : DOMString{};
        }

        bool HTMLInputElement::readOnly() const
        {
            return getAttribute ( "readonly" ) != nullptr;
        }

        uint32_t HTMLInputElement::size() const
        {
            const DOMString* attribute = getAttribute ( "size" );
            if ( !attribute )
            {
                return 20u;
            }
            const long parsed = std::strtol ( attribute->c_str(), nullptr, 10 );
            return parsed > 0 ? static_cast<uint32_t> ( parsed ) : 20u;
        }

        int32_t HTMLInputElement::maxLength() const
        {
            const DOMString* attribute = getAttribute ( "maxlength" );
            if ( !attribute )
            {
                return -1;
            }
            const long parsed = std::strtol ( attribute->c_str(), nullptr, 10 );
            return parsed >= 0 ? static_cast<int32_t> ( parsed ) : -1;
        }

        size_t HTMLInputElement::caretPosition() const
        {
            return mCaret;
        }

        double HTMLInputElement::min() const
        {
            return ParseNumber ( getAttribute ( "min" ), 0.0 );
        }

        double HTMLInputElement::max() const
        {
            // A max below min collapses the range onto min rather than
            // producing a reversed slider.
            return std::max ( min(), ParseNumber ( getAttribute ( "max" ), 100.0 ) );
        }

        double HTMLInputElement::step() const
        {
            const DOMString* attribute = getAttribute ( "step" );
            if ( attribute && EqualsIgnoreCase ( *attribute, "any" ) )
            {
                return 0.0;
            }
            const double parsed = ParseNumber ( attribute, 1.0 );
            return parsed > 0.0 ? parsed : 1.0;
        }

        double HTMLInputElement::valueAsNumber() const
        {
            const double lower = min();
            const double upper = max();
            // The default value is the midpoint of the range, per the
            // value sanitization algorithm for type=range.
            const double raw = ParseNumber ( &mValue, lower + ( upper - lower ) * 0.5 );
            const double clamped = std::clamp ( raw, lower, upper );
            const double granularity = step();
            if ( granularity <= 0.0 )
            {
                return clamped;
            }
            const double snapped = lower + std::round ( ( clamped - lower ) / granularity ) * granularity;
            return std::clamp ( snapped, lower, upper );
        }

        void HTMLInputElement::setValueAsNumber ( double aValue )
        {
            if ( SetRangeValue ( aValue ) )
            {
                RequestRedraw();
            }
        }

        bool HTMLInputElement::SetRangeValue ( double aValue )
        {
            const DOMString previous = mValue;
            mValue = NumberToString ( aValue );
            mValue = NumberToString ( valueAsNumber() );
            return mValue != previous;
        }

        DOMString HTMLInputElement::value() const
        {
            if ( mType == Type::Checkbox || mType == Type::Radio )
            {
                const DOMString* attribute = getAttribute ( "value" );
                return attribute ? *attribute : DOMString{"on"};
            }
            if ( mType == Type::Range )
            {
                return NumberToString ( valueAsNumber() );
            }
            return mValue;
        }

        void HTMLInputElement::setValue ( const DOMString& aValue )
        {
            if ( mValue == aValue )
            {
                return;
            }
            mValue = aValue;
            mCaret = std::min ( mCaret, mValue.size() );
            RequestRedraw();
        }

        void HTMLInputElement::Reset()
        {
            mValue = defaultValue();
            mCaret = mValue.size();
            const bool default_checked = defaultChecked();
            if ( mChecked != default_checked )
            {
                mChecked = default_checked;
                ReselectCSS();
            }
            RequestRedraw();
        }

        void HTMLInputElement::AppendFormData ( FormData& aFormData ) const
        {
            const DOMString control_name = name();
            if ( control_name.empty() || isDisabled() ||
                 mType == Type::Unsupported || IsButtonType ( mType ) )
            {
                return;
            }
            if ( ( mType == Type::Checkbox || mType == Type::Radio ) && !mChecked )
            {
                return;
            }
            aFormData.emplace_back ( control_name, value() );
        }

        void HTMLInputElement::Activate()
        {
            if ( isDisabled() )
            {
                return;
            }
            switch ( mType )
            {
            case Type::Checkbox:
                mChecked = !mChecked;
                ReselectCSS();
                RequestRedraw();
                break;
            case Type::Radio:
                if ( mChecked )
                {
                    return;
                }
                mChecked = true;
                ClearRadioGroup();
                ReselectCSS();
                RequestRedraw();
                break;
            case Type::Submit:
                if ( HTMLFormElement * owner = form() )
                {
                    owner->submit();
                }
                return;
            case Type::Reset:
                if ( HTMLFormElement * owner = form() )
                {
                    owner->reset();
                }
                return;
            default:
                return;
            }
            Event inputEvent ( "input", EventInit{true, false, false} );
            dispatchEvent ( inputEvent );
            Event changeEvent ( "change", EventInit{true, false, false} );
            dispatchEvent ( changeEvent );
        }

        bool HTMLInputElement::IsTextEditable() const
        {
            return IsTextType ( mType ) && !readOnly() && !isDisabled();
        }

        bool HTMLInputElement::HandleKey ( const DOMString& aKey )
        {
            if ( isDisabled() )
            {
                return false;
            }
            if ( mType == Type::Range )
            {
                const double lower = min();
                const double upper = max();
                const double granularity = step() > 0.0 ? step() : ( upper - lower ) / 100.0;
                const double page = std::max ( granularity, ( upper - lower ) / 10.0 );
                double target = valueAsNumber();
                if ( aKey == "ArrowLeft" || aKey == "ArrowDown" )
                {
                    target -= granularity;
                }
                else if ( aKey == "ArrowRight" || aKey == "ArrowUp" )
                {
                    target += granularity;
                }
                else if ( aKey == "PageDown" )
                {
                    target -= page;
                }
                else if ( aKey == "PageUp" )
                {
                    target += page;
                }
                else if ( aKey == "Home" )
                {
                    target = lower;
                }
                else if ( aKey == "End" )
                {
                    target = upper;
                }
                else
                {
                    return false;
                }
                if ( SetRangeValue ( target ) )
                {
                    RequestRedraw();
                    Event inputEvent ( "input", EventInit{true, false, false} );
                    dispatchEvent ( inputEvent );
                    Event changeEvent ( "change", EventInit{true, false, false} );
                    dispatchEvent ( changeEvent );
                }
                return true;
            }
            if ( mType == Type::Checkbox || mType == Type::Radio || IsButtonType ( mType ) )
            {
                if ( aKey == " " || ( aKey == "Enter" && IsButtonType ( mType ) ) )
                {
                    Activate();
                    return true;
                }
                return false;
            }
            if ( !IsTextType ( mType ) )
            {
                return false;
            }
            if ( aKey == "Enter" )
            {
                // Implicit submission: a lone text field submits its form.
                if ( HTMLFormElement * owner = form() )
                {
                    owner->submit();
                    return true;
                }
                return false;
            }
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
                mCaret = 0;
                RequestRedraw();
                return true;
            }
            if ( aKey == "End" )
            {
                mCaret = mValue.size();
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
            else if ( IsPrintableKey ( aKey ) )
            {
                const int32_t limit = maxLength();
                if ( limit >= 0 && CountCodePoints ( mValue ) >= static_cast<size_t> ( limit ) )
                {
                    return true;
                }
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

        bool HTMLInputElement::isChecked() const
        {
            return mChecked;
        }

        void HTMLInputElement::RangeTrackGeometry ( double& aStartX, double& aTrackWidth,
                double& aThumbRadius ) const
        {
            const LayoutBox& box = GetLayoutBox();
            aThumbRadius = std::min ( 8.0, std::max ( 3.0, box.contentHeight * 0.5 ) );
            aStartX = box.contentX + aThumbRadius;
            aTrackWidth = std::max ( 0.0, box.contentWidth - aThumbRadius * 2.0 );
        }

        bool HTMLInputElement::HandlePointerDrag ( double aX, double )
        {
            if ( mType != Type::Range || isDisabled() )
            {
                return false;
            }
            double start_x{};
            double track_width{};
            double thumb_radius{};
            RangeTrackGeometry ( start_x, track_width, thumb_radius );
            if ( track_width <= 0.0 )
            {
                return false;
            }
            const double fraction = std::clamp ( ( aX - start_x ) / track_width, 0.0, 1.0 );
            const double lower = min();
            if ( !SetRangeValue ( lower + fraction * ( max() - lower ) ) )
            {
                return true;
            }
            mRangeDirty = true;
            RequestRedraw();
            Event inputEvent ( "input", EventInit{true, false, false} );
            dispatchEvent ( inputEvent );
            return true;
        }

        void HTMLInputElement::EndPointerDrag()
        {
            if ( !mRangeDirty )
            {
                return;
            }
            mRangeDirty = false;
            Event changeEvent ( "change", EventInit{true, false, false} );
            dispatchEvent ( changeEvent );
        }

        bool HTMLInputElement::GetIntrinsicContentSize ( float& aWidth, float& aHeight ) const
        {
            if ( mType == Type::Hidden || mType == Type::Unsupported )
            {
                aWidth = 0.0f;
                aHeight = 0.0f;
                return true;
            }
            if ( mType == Type::Range )
            {
                // Matches the de-facto default slider metrics browsers use.
                aWidth  = 129.0f;
                aHeight = 21.0f;
                return true;
            }

            const FontSpec font = GetFontSpec();
            PangoTextLayout layout;
            layout.SetFontFamily ( font.family );
            layout.SetFontSize   ( font.size );
            layout.SetFontWeight ( font.weight );
            layout.SetFontStyle  ( font.style );

            if ( IsButtonType ( mType ) )
            {
                layout.SetText ( value().empty() ? DefaultButtonLabel() : value() );
                aWidth  = static_cast<float> ( layout.GetTextWidth() );
                aHeight = static_cast<float> ( layout.GetTextHeight() );
                return true;
            }
            if ( mType == Type::Checkbox || mType == Type::Radio )
            {
                // The UA sheet pins these to 13x13; report the same so
                // an author overriding only one axis still gets a
                // sensible box on the other.
                aWidth  = 13.0f;
                aHeight = 13.0f;
                return true;
            }
            // Text fields size to `size` characters of the "0" glyph,
            // the same average-advance heuristic browsers use.
            layout.SetText ( "0" );
            aWidth  = static_cast<float> ( layout.GetTextWidth() * size() );
            aHeight = static_cast<float> ( layout.GetTextHeight() );
            return true;
        }

        void HTMLInputElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            HTMLFormControlElement::onAttributeChanged ( aName, aValue );
            if ( aName == "type" )
            {
                mType = ParseType ( aValue );
                ReselectCSS();
            }
            else if ( aName == "value" )
            {
                mValue = aValue;
                mCaret = mValue.size();
            }
            else if ( aName == "checked" )
            {
                mChecked = true;
                ReselectCSS();
            }
            else if ( aName == "disabled" || aName == "readonly" )
            {
                ReselectCSS();
            }
        }

        void HTMLInputElement::ClearRadioGroup()
        {
            const DOMString group = name();
            if ( group.empty() )
            {
                return;
            }
            Node* scope = form();
            if ( !scope )
            {
                scope = ownerDocument();
            }
            if ( !scope )
            {
                return;
            }
            scope->TraverseDepthFirstPreOrder ( [this, &group] ( Node & aNode )
            {
                auto* peer = dynamic_cast<HTMLInputElement*> ( &aNode );
                if ( peer && peer != this && peer->mType == Type::Radio &&
                     peer->name() == group && peer->form() == form() )
                {
                    if ( peer->mChecked )
                    {
                        peer->mChecked = false;
                        peer->ReselectCSS();
                    }
                }
            } );
        }

        DOMString HTMLInputElement::DefaultButtonLabel() const
        {
            switch ( mType )
            {
            case Type::Submit:
                return DOMString{"Submit"};
            case Type::Reset:
                return DOMString{"Reset"};
            default:
                return DOMString{};
            }
        }

        void HTMLInputElement::DrawStart ( Canvas& aCanvas ) const
        {
            if ( mType == Type::Hidden || mType == Type::Unsupported )
            {
                return;
            }
            const LayoutBox& box = GetLayoutBox();
            if ( box.width <= 0.0f || box.height <= 0.0f )
            {
                return;
            }

            if ( mType == Type::Radio )
            {
                PaintRadio ( aCanvas );
                return;
            }

            PaintBox ( aCanvas );

            if ( mType == Type::Checkbox )
            {
                PaintCheckMark ( aCanvas );
                return;
            }
            if ( mType == Type::Range )
            {
                PaintRange ( aCanvas );
                return;
            }
            if ( IsButtonType ( mType ) )
            {
                PaintButtonLabel ( aCanvas );
                return;
            }
            PaintTextField ( aCanvas );
        }

        void HTMLInputElement::PaintRange ( Canvas& aCanvas ) const
        {
            const LayoutBox& box = GetLayoutBox();
            double start_x{};
            double track_width{};
            double thumb_radius{};
            RangeTrackGeometry ( start_x, track_width, thumb_radius );
            if ( track_width <= 0.0 )
            {
                return;
            }

            const double lower = min();
            const double span = max() - lower;
            const double fraction = span > 0.0 ? ( valueAsNumber() - lower ) / span : 0.0;
            const double thumb_x = start_x + fraction * track_width;
            const double center_y = box.contentY + box.contentHeight * 0.5;
            const double track_height = std::max ( 2.0, thumb_radius * 0.4 );

            ColorAttr accent{};
            if ( !ResolveTextColor ( accent ) )
            {
                accent = ColorAttr{ Color{ 0xFF000000u } };
            }
            const ColorAttr groove{ Color{ isDisabled() ? 0xFFD0D0D0u : 0xFFB4B4B4u } };

            const ColorAttr previous_fill = aCanvas.GetFillColor();

            aCanvas.SetFillColor ( groove );
            FillRect ( aCanvas, box.contentX, center_y - track_height * 0.5,
                       box.contentX + box.contentWidth, center_y + track_height * 0.5 );

            aCanvas.SetFillColor ( accent );
            FillRect ( aCanvas, box.contentX, center_y - track_height * 0.5,
                       thumb_x, center_y + track_height * 0.5 );
            FillEllipse ( aCanvas, thumb_x, center_y, thumb_radius, thumb_radius );
            aCanvas.SetFillColor ( previous_fill );

            // The track and thumb leave most of the widget box untouched;
            // stamping it keeps the slider clickable end to end.
            PaintHitArea ( aCanvas );
        }

        void HTMLInputElement::PaintRadio ( Canvas& aCanvas ) const
        {
            css_computed_style* style = MutableStyleOf ( *this );
            if ( !style )
            {
                return;
            }
            const LayoutBox& box = GetLayoutBox();
            const double center_x = box.x + box.width  * 0.5;
            const double center_y = box.y + box.height * 0.5;
            const double radius = std::min ( box.width, box.height ) * 0.5;
            if ( radius <= 0.0 )
            {
                return;
            }
            const ColorAttr previous_fill = aCanvas.GetFillColor();

            css_color border_color{};
            if ( css_computed_border_top_color ( style, &border_color ) !=
                 CSS_BORDER_COLOR_COLOR || border_color == 0 )
            {
                css_computed_color ( style, &border_color );
            }
            aCanvas.SetFillColor ( ColorAttr{ Color{ static_cast<uint32_t> ( border_color ) } } );
            FillEllipse ( aCanvas, center_x, center_y, radius, radius );

            css_color background{};
            if ( css_computed_background_color ( style, &background ) !=
                 CSS_BACKGROUND_COLOR_COLOR || background == 0 )
            {
                background = 0xFFFFFFFF;
            }
            aCanvas.SetFillColor ( ColorAttr{ Color{ static_cast<uint32_t> ( background ) } } );
            FillEllipse ( aCanvas, center_x, center_y, radius - 1.0, radius - 1.0 );

            if ( mChecked )
            {
                ColorAttr dot{};
                if ( ResolveTextColor ( dot ) )
                {
                    aCanvas.SetFillColor ( dot );
                    const double dot_radius = std::max ( 1.0, radius - 3.0 );
                    FillEllipse ( aCanvas, center_x, center_y, dot_radius, dot_radius );
                }
            }
            aCanvas.SetFillColor ( previous_fill );
        }

        void HTMLInputElement::PaintCheckMark ( Canvas& aCanvas ) const
        {
            if ( !mChecked )
            {
                return;
            }
            ColorAttr mark{};
            if ( !ResolveTextColor ( mark ) )
            {
                return;
            }
            const LayoutBox& box = GetLayoutBox();
            const double x = box.contentX;
            const double y = box.contentY;
            const double w = box.contentWidth;
            const double h = box.contentHeight;
            if ( w <= 0.0 || h <= 0.0 )
            {
                return;
            }
            const ColorAttr previous_fill = aCanvas.GetFillColor();
            aCanvas.SetFillColor ( mark );
            // A tick built from two overlapping quadrilaterals: the
            // short down-stroke and the long up-stroke.
            const double thickness = std::max ( 1.5, std::min ( w, h ) * 0.18 );
            const double ax = x + w * 0.18;
            const double ay = y + h * 0.50;
            const double bx = x + w * 0.42;
            const double by = y + h * 0.74;
            const double cx = x + w * 0.84;
            const double cy = y + h * 0.24;
            FillStroke ( aCanvas, ax, ay, bx, by, thickness );
            FillStroke ( aCanvas, bx, by, cx, cy, thickness );
            aCanvas.SetFillColor ( previous_fill );
        }

        void HTMLInputElement::PaintButtonLabel ( Canvas& aCanvas ) const
        {
            DOMString label = value();
            if ( label.empty() )
            {
                label = DefaultButtonLabel();
            }
            if ( label.empty() )
            {
                return;
            }
            ColorAttr text_color{};
            if ( !ResolveTextColor ( text_color ) )
            {
                return;
            }
            const LayoutBox& box = GetLayoutBox();
            const FontSpec font = GetFontSpec();
            PangoTextLayout layout;
            layout.SetFontFamily ( font.family );
            layout.SetFontSize   ( font.size );
            layout.SetFontWeight ( font.weight );
            layout.SetFontStyle  ( font.style );
            layout.SetText ( label );

            const double text_width = layout.GetTextWidth();
            const double x = box.contentX +
                             std::max ( 0.0, ( box.contentWidth - text_width ) * 0.5 );
            const double baseline = box.contentY +
                                    std::max ( 0.0, ( box.contentHeight - layout.GetTextHeight() ) * 0.5 ) +
                                    layout.GetBaseline();

            const ColorAttr previous_fill = aCanvas.GetFillColor();
            aCanvas.SetFillColor ( text_color );
            aCanvas.DrawText ( label, x, baseline, font.family, font.size, font.weight, font.style );
            aCanvas.SetFillColor ( previous_fill );
        }

        void HTMLInputElement::PaintTextField ( Canvas& aCanvas ) const
        {
            const LayoutBox& box = GetLayoutBox();
            if ( box.contentWidth <= 0.0f || box.contentHeight <= 0.0f )
            {
                return;
            }
            const DOMString rendered = RenderedValue();
            const bool showing_placeholder = rendered.empty();
            const DOMString text = showing_placeholder ? placeholder() : rendered;

            const FontSpec font = GetFontSpec();
            PangoTextLayout layout;
            layout.SetFontFamily ( font.family );
            layout.SetFontSize   ( font.size );
            layout.SetFontWeight ( font.weight );
            layout.SetFontStyle  ( font.style );
            layout.SetText ( text.empty() ? DOMString{"0"} : text );

            const double line_height = layout.GetTextHeight();
            const double top = box.contentY +
                               std::max ( 0.0, ( box.contentHeight - line_height ) * 0.5 );
            const double baseline = top + layout.GetBaseline();

            const ColorAttr previous_fill = aCanvas.GetFillColor();
            aCanvas.Save();
            aCanvas.SetClipRect ( box.contentX, box.contentY,
                                  box.contentWidth, box.contentHeight );
            if ( !text.empty() )
            {
                ColorAttr text_color{};
                if ( ResolveTextColor ( text_color ) )
                {
                    if ( showing_placeholder )
                    {
                        // No ::placeholder pseudo-element support yet;
                        // fall back to the conventional grey.
                        text_color = ColorAttr{ Color{ 0xFF767676u } };
                    }
                    aCanvas.SetFillColor ( text_color );
                    aCanvas.DrawText ( text, box.contentX, baseline,
                                       font.family, font.size, font.weight, font.style );
                }
            }
            if ( isFocus() && !isDisabled() && !readOnly() )
            {
                const size_t caret_bytes = showing_placeholder
                                           ? 0u
                                           : CaretOffsetInRenderedValue();
                const double caret_x = box.contentX +
                                       ( showing_placeholder
                                         ? 0.0
                                         : layout.GetCharOffsetX ( static_cast<int32_t> ( caret_bytes ) ) );
                PaintCaret ( aCanvas, caret_x, top, line_height );
            }
            aCanvas.Restore();
            aCanvas.SetFillColor ( previous_fill );
        }

        DOMString HTMLInputElement::RenderedValue() const
        {
            if ( mType != Type::Password )
            {
                return mValue;
            }
            DOMString masked;
            masked.reserve ( CountCodePoints ( mValue ) * ( sizeof ( kPasswordBullet ) - 1 ) );
            for ( size_t i = 0; i < CountCodePoints ( mValue ); ++i )
            {
                masked.append ( kPasswordBullet );
            }
            return masked;
        }

        size_t HTMLInputElement::CaretOffsetInRenderedValue() const
        {
            if ( mType != Type::Password )
            {
                return mCaret;
            }
            return CountCodePoints ( mValue.substr ( 0, mCaret ) ) *
                   ( sizeof ( kPasswordBullet ) - 1 );
        }

        void HTMLInputElement::FillStroke ( Canvas& aCanvas,
                                            double aX0, double aY0,
                                            double aX1, double aY1,
                                            double aThickness )
        {
            const double dx = aX1 - aX0;
            const double dy = aY1 - aY0;
            const double length = std::sqrt ( dx * dx + dy * dy );
            if ( length <= 0.0 )
            {
                return;
            }
            const double nx = -dy / length * aThickness * 0.5;
            const double ny =  dx / length * aThickness * 0.5;
            const std::array<double, 8> points
            {
                aX0 + nx, aY0 + ny,
                aX1 + nx, aY1 + ny,
                aX1 - nx, aY1 - ny,
                aX0 - nx, aY0 - ny,
            };
            FillPolygon ( aCanvas, points.data(), 4 );
        }
    }
}
