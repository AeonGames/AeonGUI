/*
Copyright (C) 2019,2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include <iostream>
#include "aeongui/dom/SVGUseElement.hpp"
#include "aeongui/dom/SVGGeometryElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include "aeongui/Matrix2x3.hpp"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        SVGUseElement::SVGUseElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGraphicsElement {aTagName, std::move ( aAttributes ), aParent}
        {
            ParseAttributes();
        }
        SVGUseElement::~SVGUseElement() = default;

        void SVGUseElement::ParseAttributes()
        {
            auto it = mAttributes.find ( "href" );
            if ( it == mAttributes.end() )
            {
                it = mAttributes.find ( "xlink:href" );
            }
            if ( it != mAttributes.end() )
            {
                mHref = it->second;
            }
            if ( mAttributes.find ( "x" ) != mAttributes.end() )
            {
                mX = std::stod ( mAttributes.at ( "x" ) );
            }
            if ( mAttributes.find ( "y" ) != mAttributes.end() )
            {
                mY = std::stod ( mAttributes.at ( "y" ) );
            }
        }

        void SVGUseElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "href" || aName == "xlink:href" || aName == "x" || aName == "y" )
            {
                ParseAttributes();
            }
        }

        void SVGUseElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );

            if ( mX != 0.0 || mY != 0.0 )
            {
                aCanvas.Transform ( Matrix2x3{1, 0, 0, 1, mX, mY} );
            }

            if ( mHref.empty() )
            {
                return;
            }

            std::string id = mHref;
            if ( !id.empty() && id[0] == '#' )
            {
                id = id.substr ( 1 );
            }

            const Node* node = this;
            while ( node->parentNode() )
            {
                node = node->parentNode();
            }
            const Document* document = static_cast<const Document*> ( node );

            Element* referenced = document->getElementById ( id );
            if ( !referenced )
            {
                return;
            }

            const SVGGeometryElement* geomElement = dynamic_cast<const SVGGeometryElement*> ( referenced );
            if ( !geomElement )
            {
                return;
            }

            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                ApplyCSSPaintProperties ( aCanvas, *this, results->styles[CSS_PSEUDO_ELEMENT_NONE] );
            }

            aCanvas.Draw ( geomElement->GetPath() );
        }
    }
}