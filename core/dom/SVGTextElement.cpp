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
#include "aeongui/dom/SVGTextElement.hpp"
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
        SVGTextElement::SVGTextElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) :
            SVGTextPositioningElement { aTagName, std::move ( aAttributes ), aParent }
        {
        }
        SVGTextElement::~SVGTextElement() = default;

        void SVGTextElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGTextPositioningElement::DrawStart ( aCanvas );
            css_select_results* results{ GetComputedStyles() };
            if ( !results || !results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                return;
            }
            css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];

            ApplyCSSPaintProperties ( aCanvas, *this, style );
            ApplyChildPaintAnimations ( aCanvas );

            // Font properties from CSS.
            std::string fontFamily = GetCSSFontFamily ( style );
            double fontSize = GetCSSFontSize ( style );
            int fontWeight = GetCSSFontWeight ( style );
            int fontStyle = GetCSSFontStyle ( style );

            // Starting position from SVGTextPositioningElement x/y attributes.
            double posX = 0.0;
            double posY = 0.0;
            if ( x().baseVal().numberOfItems() > 0 )
            {
                const auto& xLen = x().baseVal().getItem ( 0 );
                posX = static_cast<double> ( xLen.value() );
                if ( xLen.unitType() == SVGLengthType::PERCENTAGE )
                {
                    posX = static_cast<double> ( xLen.valueInSpecifiedUnits() ) * aCanvas.GetViewportWidth() / 100.0;
                }
            }
            if ( y().baseVal().numberOfItems() > 0 )
            {
                const auto& yLen = y().baseVal().getItem ( 0 );
                posY = static_cast<double> ( yLen.value() );
                if ( yLen.unitType() == SVGLengthType::PERCENTAGE )
                {
                    posY = static_cast<double> ( yLen.valueInSpecifiedUnits() ) * aCanvas.GetViewportHeight() / 100.0;
                }
            }

            // text-anchor adjustment: measure total text width and offset posX.
            uint8_t textAnchor = css_computed_text_anchor ( style );
            if ( textAnchor == CSS_TEXT_ANCHOR_MIDDLE || textAnchor == CSS_TEXT_ANCHOR_END )
            {
                double totalWidth = 0.0;
                for ( const auto& child : childNodes() )
                {
                    if ( child->nodeType() == Node::TEXT_NODE )
                    {
                        const Text* textNode = static_cast<const Text*> ( child.get() );
                        std::string text = textNode->wholeText();
                        if ( !text.empty() )
                        {
                            totalWidth += aCanvas.MeasureText ( text, fontFamily, fontSize, fontWeight, fontStyle );
                        }
                    }
                }
                if ( textAnchor == CSS_TEXT_ANCHOR_MIDDLE )
                {
                    posX -= totalWidth / 2.0;
                }
                else
                {
                    posX -= totalWidth;
                }
            }

            // Render direct text children.  For tspan children the
            // recursive DrawStart call will handle them.
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
