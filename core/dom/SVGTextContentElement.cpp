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
#include "aeongui/dom/SVGTextContentElement.hpp"
#include "aeongui/dom/Text.hpp"
#include "aeongui/dom/DOMException.hpp"
#include <libcss/libcss.h>
#include <algorithm>
#include <cmath>

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextContentElement::SVGTextContentElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGraphicsElement { aTagName, std::move ( aAttributes ), aParent }
        {
        }
        SVGTextContentElement::~SVGTextContentElement() = default;
        const SVGAnimatedLength& SVGTextContentElement::textLength() const
        {
            return mTextLength;
        }

        const SVGAnimatedEnumeration& SVGTextContentElement::lengthAdjust() const
        {
            return mLengthAdjust;
        }

        PangoTextLayout& SVGTextContentElement::GetTextLayout() const
        {
            return mTextLayout;
        }

        void SVGTextContentElement::syncTextLayout() const
        {
            std::string text = getTextContent();
            mTextLayout.SetText ( text );

            // Apply font properties from CSS computed styles if available.
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];

                // Font family
                lwc_string** names = nullptr;
                uint8_t family = css_computed_font_family ( style, &names );
                if ( names && names[0] )
                {
                    mTextLayout.SetFontFamily ( std::string ( lwc_string_data ( names[0] ), lwc_string_length ( names[0] ) ) );
                }
                else
                {
                    switch ( family )
                    {
                    case CSS_FONT_FAMILY_SERIF:
                        mTextLayout.SetFontFamily ( "serif" );
                        break;
                    case CSS_FONT_FAMILY_MONOSPACE:
                        mTextLayout.SetFontFamily ( "monospace" );
                        break;
                    case CSS_FONT_FAMILY_CURSIVE:
                        mTextLayout.SetFontFamily ( "cursive" );
                        break;
                    case CSS_FONT_FAMILY_FANTASY:
                        mTextLayout.SetFontFamily ( "fantasy" );
                        break;
                    default:
                        mTextLayout.SetFontFamily ( "sans-serif" );
                        break;
                    }
                }

                // Font size
                css_fixed length{};
                css_unit unit{};
                if ( css_computed_font_size ( style, &length, &unit ) == CSS_FONT_SIZE_DIMENSION )
                {
                    mTextLayout.SetFontSize ( FIXTOFLT ( length ) );
                }

                // Font weight
                uint8_t w = css_computed_font_weight ( style );
                int weight = 400;
                switch ( w )
                {
                case CSS_FONT_WEIGHT_100:
                    weight = 100;
                    break;
                case CSS_FONT_WEIGHT_200:
                    weight = 200;
                    break;
                case CSS_FONT_WEIGHT_300:
                    weight = 300;
                    break;
                case CSS_FONT_WEIGHT_500:
                    weight = 500;
                    break;
                case CSS_FONT_WEIGHT_600:
                    weight = 600;
                    break;
                case CSS_FONT_WEIGHT_700:
                case CSS_FONT_WEIGHT_BOLD:
                    weight = 700;
                    break;
                case CSS_FONT_WEIGHT_800:
                    weight = 800;
                    break;
                case CSS_FONT_WEIGHT_900:
                    weight = 900;
                    break;
                default:
                    break;
                }
                mTextLayout.SetFontWeight ( weight );

                // Font style
                uint8_t s = css_computed_font_style ( style );
                int fontStyle = 0;
                if ( s == CSS_FONT_STYLE_ITALIC )
                {
                    fontStyle = 1;
                }
                else if ( s == CSS_FONT_STYLE_OBLIQUE )
                {
                    fontStyle = 2;
                }
                mTextLayout.SetFontStyle ( fontStyle );
            }
        }

        std::string SVGTextContentElement::getTextContent() const
        {
            std::string textContent;

            for ( const auto& child : this->childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child );
                    textContent += textNode->wholeText();
                }
                else if ( child->nodeType() == Node::ELEMENT_NODE )
                {
                    const SVGTextContentElement* childElement = dynamic_cast<const SVGTextContentElement*> ( child );
                    if ( childElement )
                    {
                        textContent += childElement->getTextContent();
                    }
                }
            }

            return textContent;
        }

        long SVGTextContentElement::getNumberOfChars() const
        {
            std::string textContent = getTextContent();
            return static_cast<long> ( textContent.length() );
        }

        float SVGTextContentElement::getComputedTextLength() const
        {
            syncTextLayout();
            return static_cast<float> ( mTextLayout.GetTextWidth() );
        }

        float SVGTextContentElement::getSubStringLength ( long start, long end ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( start < 0 )
            {
                throw DOMIndexSizeError ( "Start index cannot be negative" );
            }

            if ( start >= totalChars )
            {
                return 0.0f;
            }

            long actualEnd = std::min ( end, totalChars );
            if ( actualEnd <= start )
            {
                return 0.0f;
            }

            // Measure full text up to actualEnd and subtract measurement up to start.
            syncTextLayout();
            double endX = mTextLayout.GetCharOffsetX ( actualEnd );
            double startX = mTextLayout.GetCharOffsetX ( start );
            return static_cast<float> ( endX - startX );
        }

        DOMPoint SVGTextContentElement::getStartPositionOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            syncTextLayout();
            float xPos = static_cast<float> ( mTextLayout.GetCharOffsetX ( index ) );
            return DOMPoint ( xPos, 0.0f, 0.0f, 1.0f );
        }

        DOMPoint SVGTextContentElement::getEndPositionOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            syncTextLayout();
            float xPos = static_cast<float> ( mTextLayout.GetCharOffsetX ( index + 1 ) );
            return DOMPoint ( xPos, 0.0f, 0.0f, 1.0f );
        }

        DOMRect SVGTextContentElement::getExtentOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            syncTextLayout();
            float xStart = static_cast<float> ( mTextLayout.GetCharOffsetX ( index ) );
            float xEnd = static_cast<float> ( mTextLayout.GetCharOffsetX ( index + 1 ) );
            float charWidth = xEnd - xStart;
            float height = static_cast<float> ( mTextLayout.GetTextHeight() );

            return DOMRect ( xStart, -height, charWidth, height );
        }

        float SVGTextContentElement::getRotationOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            return 0.0f;
        }

        long SVGTextContentElement::getCharNumAtPosition ( const DOMPoint& point ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( totalChars == 0 )
            {
                return -1;
            }

            syncTextLayout();
            // Walk character positions to find hit.
            for ( long i = 0; i < totalChars; ++i )
            {
                float xStart = static_cast<float> ( mTextLayout.GetCharOffsetX ( i ) );
                float xEnd = static_cast<float> ( mTextLayout.GetCharOffsetX ( i + 1 ) );
                if ( point.x() >= xStart && point.x() < xEnd )
                {
                    return i;
                }
            }

            return -1;
        }

    }
}
