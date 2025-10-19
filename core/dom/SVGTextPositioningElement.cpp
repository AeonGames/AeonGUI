/*
Copyright (C) 2020,2024,2025 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGTextPositioningElement.hpp"
#include "aeongui/dom/SVGLength.hpp"
#include <sstream>
#include <regex>

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextPositioningElement::SVGTextPositioningElement ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGTextContentElement { aTagName, aAttributes, aParent }
        {
            // Parse positioning attributes from the attribute map
            parsePositioningAttributes ( aAttributes );
        }

        SVGTextPositioningElement::~SVGTextPositioningElement() = default;

        const SVGAnimatedLengthList& SVGTextPositioningElement::x() const
        {
            return mX;
        }

        const SVGAnimatedLengthList& SVGTextPositioningElement::y() const
        {
            return mY;
        }

        const SVGAnimatedLengthList& SVGTextPositioningElement::dx() const
        {
            return mDx;
        }

        const SVGAnimatedLengthList& SVGTextPositioningElement::dy() const
        {
            return mDy;
        }

        const SVGAnimatedNumberList& SVGTextPositioningElement::rotate() const
        {
            return mRotate;
        }

        void SVGTextPositioningElement::parsePositioningAttributes ( const AttributeMap& aAttributes )
        {
            // Parse x attribute
            auto it = aAttributes.find ( "x" );
            if ( it != aAttributes.end() )
            {
                parseLengthList ( it->second, mX.baseVal() );
            }

            // Parse y attribute
            it = aAttributes.find ( "y" );
            if ( it != aAttributes.end() )
            {
                parseLengthList ( it->second, mY.baseVal() );
            }

            // Parse dx attribute
            it = aAttributes.find ( "dx" );
            if ( it != aAttributes.end() )
            {
                parseLengthList ( it->second, mDx.baseVal() );
            }

            // Parse dy attribute
            it = aAttributes.find ( "dy" );
            if ( it != aAttributes.end() )
            {
                parseLengthList ( it->second, mDy.baseVal() );
            }

            // Parse rotate attribute
            it = aAttributes.find ( "rotate" );
            if ( it != aAttributes.end() )
            {
                parseNumberList ( it->second, mRotate.baseVal() );
            }
        }

        void SVGTextPositioningElement::parseLengthList ( const DOMString& value, SVGLengthList& lengthList )
        {
            lengthList.clear();

            // Regular expression to match length values (numbers with optional units)
            // Supports whitespace and comma separation
            std::regex lengthRegex ( R"(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(px|cm|mm|in|pt|pc|%|em|ex)?)" );
            std::sregex_iterator iter ( value.begin(), value.end(), lengthRegex );
            std::sregex_iterator end;

            for ( ; iter != end; ++iter )
            {
                const std::smatch& match = *iter;
                SVGLength length;

                // Set the length value from the parsed string
                DOMString lengthStr = match[0].str();
                length.valueAsString ( lengthStr );

                lengthList.appendItem ( length );
            }
        }

        void SVGTextPositioningElement::parseNumberList ( const DOMString& value, SVGNumberList& numberList )
        {
            numberList.clear();

            // Regular expression to match number values
            // Supports whitespace and comma separation
            std::regex numberRegex ( R"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)" );
            std::sregex_iterator iter ( value.begin(), value.end(), numberRegex );
            std::sregex_iterator end;

            for ( ; iter != end; ++iter )
            {
                const std::smatch& match = *iter;
                float number = std::stof ( match[0].str() );
                numberList.appendItem ( number );
            }
        }
    }
}
