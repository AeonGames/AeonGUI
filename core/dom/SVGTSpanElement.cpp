/*
Copyright (C) 2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGTSpanElement.hpp"
#include "aeongui/dom/Text.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include <libcss/libcss.h>
#include <string>
namespace AeonGUI
{
    namespace DOM
    {
        SVGTSpanElement::SVGTSpanElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGTextPositioningElement { aTagName, std::move ( aAttributes ), aParent }
        {
        }
        SVGTSpanElement::~SVGTSpanElement() = default;

        void SVGTSpanElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGTextPositioningElement::DrawStart ( aCanvas );
            css_select_results* results{ GetComputedStyles() };
            if ( !results || !results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                return;
            }
            css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];

            ApplyCSSPaintProperties ( aCanvas, style );

            std::string fontFamily = GetCSSFontFamily ( style );
            double fontSize = GetCSSFontSize ( style );
            int fontWeight = GetCSSFontWeight ( style );
            int fontStyle = GetCSSFontStyle ( style );

            // Position: tspan may have absolute x/y and/or relative dx/dy.
            double posX = 0.0;
            double posY = 0.0;
            if ( x().baseVal().numberOfItems() > 0 )
            {
                posX = static_cast<double> ( x().baseVal().getItem ( 0 ).value() );
            }
            if ( y().baseVal().numberOfItems() > 0 )
            {
                posY = static_cast<double> ( y().baseVal().getItem ( 0 ).value() );
            }
            if ( dx().baseVal().numberOfItems() > 0 )
            {
                posX += static_cast<double> ( dx().baseVal().getItem ( 0 ).value() );
            }
            if ( dy().baseVal().numberOfItems() > 0 )
            {
                posY += static_cast<double> ( dy().baseVal().getItem ( 0 ).value() );
            }

            for ( const auto& child : childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child.get() );
                    std::string text = textNode->wholeText();
                    if ( !text.empty() )
                    {
                        aCanvas.DrawText ( text, posX, posY, fontFamily, fontSize, fontWeight, fontStyle );
                        posX += aCanvas.MeasureText ( text, fontFamily, fontSize, fontWeight, fontStyle );
                    }
                }
            }
        }
    }
}
