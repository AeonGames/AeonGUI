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
#include "aeongui/StyleSheet.hpp"
#ifdef AEONGUI_USE_SKIA
#include "SkiaTextLayout.hpp"
#else
#include "PangoTextLayout.hpp"
#endif
#include <libcss/libcss.h>
#include <algorithm>
#include <cmath>

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextContentElement::SVGTextContentElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGraphicsElement { aTagName, std::move ( aAttributes ), aParent },
#ifdef AEONGUI_USE_SKIA
            mTextLayout { std::make_unique<SkiaTextLayout>() }
#else
            mTextLayout { std::make_unique<PangoTextLayout>() }
#endif
        {
        }
        SVGTextContentElement::~SVGTextContentElement() = default;
        const SVGAnimatedLength & SVGTextContentElement::textLength() const
        {
            return mTextLength;
        }

        const SVGAnimatedEnumeration& SVGTextContentElement::lengthAdjust() const
        {
            return mLengthAdjust;
        }

        TextLayout& SVGTextContentElement::GetTextLayout() const
        {
            return *mTextLayout;
        }

        void SVGTextContentElement::syncTextLayout() const
        {
            std::string text = getTextContent();
            mTextLayout->SetText ( text );

            // Apply font properties from CSS computed styles if available.
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];
                mTextLayout->SetFontFamily ( GetCSSFontFamily ( style ) );
                mTextLayout->SetFontSize ( GetCSSFontSize ( style ) );
                mTextLayout->SetFontWeight ( GetCSSFontWeight ( style ) );
                mTextLayout->SetFontStyle ( GetCSSFontStyle ( style ) );
            }
        }

        std::string SVGTextContentElement::getTextContent() const
        {
            std::string textContent;

            for ( const auto& child : this->childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child.get() );
                    textContent += textNode->wholeText();
                }
                else if ( child->nodeType() == Node::ELEMENT_NODE )
                {
                    const SVGTextContentElement* childElement = dynamic_cast<const SVGTextContentElement*> ( child.get() );
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
            return static_cast<float> ( mTextLayout->GetTextWidth() );
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
            double endX = mTextLayout->GetCharOffsetX ( actualEnd );
            double startX = mTextLayout->GetCharOffsetX ( start );
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
            float xPos = static_cast<float> ( mTextLayout->GetCharOffsetX ( index ) );
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
            float xPos = static_cast<float> ( mTextLayout->GetCharOffsetX ( index + 1 ) );
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
            float xStart = static_cast<float> ( mTextLayout->GetCharOffsetX ( index ) );
            float xEnd = static_cast<float> ( mTextLayout->GetCharOffsetX ( index + 1 ) );
            float charWidth = xEnd - xStart;
            float height = static_cast<float> ( mTextLayout->GetTextHeight() );

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
                float xStart = static_cast<float> ( mTextLayout->GetCharOffsetX ( i ) );
                float xEnd = static_cast<float> ( mTextLayout->GetCharOffsetX ( i + 1 ) );
                if ( point.x() >= xStart && point.x() < xEnd )
                {
                    return i;
                }
            }

            return -1;
        }

    }
}
