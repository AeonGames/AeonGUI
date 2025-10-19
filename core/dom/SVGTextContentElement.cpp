/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include <algorithm>
#include <cmath>

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextContentElement::SVGTextContentElement ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGGraphicsElement { aTagName, aAttributes, aParent }
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

        std::string SVGTextContentElement::getTextContent() const
        {
            std::string textContent;

            // Traverse all child nodes and collect text content
            for ( const auto& child : this->childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child );
                    textContent += textNode->wholeText();
                }
                // For child elements, recursively get their text content
                else if ( child->nodeType() == Node::ELEMENT_NODE )
                {
                    // Cast to SVGTextContentElement if it's a text content element
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
            // This should compute the actual rendered length of the text
            // For now, return a basic approximation based on character count
            // In a real implementation, this would use font metrics and rendering context
            std::string textContent = getTextContent();

            // Basic approximation: assume average character width of 8 pixels
            // This should be replaced with actual font metrics calculation
            const float averageCharWidth = 8.0f;
            return static_cast<float> ( textContent.length() ) * averageCharWidth;
        }

        float SVGTextContentElement::getSubStringLength ( long start, long end ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            // Validate parameters according to SVG specification
            if ( start < 0 )
            {
                throw DOMIndexSizeError ( "Start index cannot be negative" );
            }

            if ( start >= totalChars )
            {
                return 0.0f; // Return 0 if start is beyond text length
            }

            // Clamp end to valid range
            long actualEnd = std::min ( end, totalChars );
            if ( actualEnd <= start )
            {
                return 0.0f; // Return 0 if end is not after start
            }

            // Calculate length of substring
            long substringLength = actualEnd - start;

            // Basic approximation: assume average character width of 8 pixels
            const float averageCharWidth = 8.0f;
            return static_cast<float> ( substringLength ) * averageCharWidth;
        }

        DOMPoint SVGTextContentElement::getStartPositionOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            // Basic approximation: assume characters are laid out horizontally
            // starting at origin with average character width of 8 pixels
            const float averageCharWidth = 8.0f;
            float x = static_cast<float> ( index ) * averageCharWidth;

            // Y position would depend on font metrics - using 0 for now
            return DOMPoint ( x, 0.0f, 0.0f, 1.0f );
        }

        DOMPoint SVGTextContentElement::getEndPositionOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            // Basic approximation: end position is start position + character width
            const float averageCharWidth = 8.0f;
            float x = static_cast<float> ( index + 1 ) * averageCharWidth;

            return DOMPoint ( x, 0.0f, 0.0f, 1.0f );
        }

        DOMRect SVGTextContentElement::getExtentOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            // Basic approximation: character extent as a rectangle
            const float averageCharWidth = 8.0f;
            const float averageCharHeight = 12.0f; // Typical font height

            float x = static_cast<float> ( index ) * averageCharWidth;
            float y = -averageCharHeight; // Negative because text baseline is typically above the origin

            return DOMRect ( x, y, averageCharWidth, averageCharHeight );
        }

        float SVGTextContentElement::getRotationOfChar ( long index ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( index < 0 || index >= totalChars )
            {
                throw DOMIndexSizeError ( "Character index out of bounds" );
            }

            // Basic implementation: assume no rotation
            // In a full implementation, this would consider text-on-path, transforms, etc.
            return 0.0f;
        }

        long SVGTextContentElement::getCharNumAtPosition ( const DOMPoint& point ) const
        {
            std::string textContent = getTextContent();
            long totalChars = static_cast<long> ( textContent.length() );

            if ( totalChars == 0 )
            {
                return -1; // No characters available
            }

            // Basic approximation: determine character based on x coordinate
            const float averageCharWidth = 8.0f;

            // Calculate which character position this point corresponds to
            long charIndex = static_cast<long> ( std::floor ( point.x() / averageCharWidth ) );

            // Clamp to valid range
            if ( charIndex < 0 )
            {
                return -1; // Before first character
            }

            if ( charIndex >= totalChars )
            {
                return -1; // After last character
            }

            return charIndex;
        }

    }
}
