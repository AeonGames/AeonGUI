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
#include <libcss/libcss.h>
#include <string>
namespace AeonGUI
{
    namespace DOM
    {
        static std::string GetCSSFontFamily ( css_computed_style* aStyle )
        {
            lwc_string** names = nullptr;
            uint8_t family = css_computed_font_family ( aStyle, &names );
            if ( names && names[0] )
            {
                return std::string ( lwc_string_data ( names[0] ), lwc_string_length ( names[0] ) );
            }
            switch ( family )
            {
            case CSS_FONT_FAMILY_SERIF:
                return "serif";
            case CSS_FONT_FAMILY_MONOSPACE:
                return "monospace";
            case CSS_FONT_FAMILY_CURSIVE:
                return "cursive";
            case CSS_FONT_FAMILY_FANTASY:
                return "fantasy";
            case CSS_FONT_FAMILY_SANS_SERIF:
            default:
                return "sans-serif";
            }
        }

        static double GetCSSFontSize ( css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t sizeType = css_computed_font_size ( aStyle, &length, &unit );
            if ( sizeType == CSS_FONT_SIZE_DIMENSION )
            {
                // @todo Convert units other than px properly.
                return FIXTOFLT ( length );
            }
            return 16.0; // Default medium size.
        }

        static int GetCSSFontWeight ( css_computed_style* aStyle )
        {
            uint8_t w = css_computed_font_weight ( aStyle );
            switch ( w )
            {
            case CSS_FONT_WEIGHT_100:
                return 100;
            case CSS_FONT_WEIGHT_200:
                return 200;
            case CSS_FONT_WEIGHT_300:
                return 300;
            case CSS_FONT_WEIGHT_400:
            case CSS_FONT_WEIGHT_NORMAL:
                return 400;
            case CSS_FONT_WEIGHT_500:
                return 500;
            case CSS_FONT_WEIGHT_600:
                return 600;
            case CSS_FONT_WEIGHT_700:
            case CSS_FONT_WEIGHT_BOLD:
                return 700;
            case CSS_FONT_WEIGHT_800:
                return 800;
            case CSS_FONT_WEIGHT_900:
                return 900;
            default:
                return 400;
            }
        }

        static int GetCSSFontStyle ( css_computed_style* aStyle )
        {
            uint8_t s = css_computed_font_style ( aStyle );
            switch ( s )
            {
            case CSS_FONT_STYLE_ITALIC:
                return 1;
            case CSS_FONT_STYLE_OBLIQUE:
                return 2;
            default:
                return 0;
            }
        }

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

            // Read fill / stroke / opacity just like SVGGeometryElement.
            css_color color{};
            css_fixed fixed{};
            css_unit unit{};
            if ( css_computed_fill ( style, &color ) != CSS_PAINT_NONE )
            {
                aCanvas.SetFillColor ( Color{color} );
                css_computed_fill_opacity ( style, &fixed );
                aCanvas.SetFillOpacity ( FIXTOFLT ( fixed ) );
            }
            if ( css_computed_stroke ( style, &color ) )
            {
                aCanvas.SetStrokeColor ( Color{color} );
                css_computed_stroke_opacity ( style, &fixed );
                aCanvas.SetStrokeOpacity ( FIXTOFLT ( fixed ) );
                css_computed_stroke_width ( style, &fixed, &unit );
                aCanvas.SetStrokeWidth ( FIXTOFLT ( fixed ) );
            }
            css_computed_opacity ( style, &fixed );
            aCanvas.SetOpacity ( FIXTOFLT ( fixed ) );

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
                posX = static_cast<double> ( x().baseVal().getItem ( 0 ).value() );
            }
            if ( y().baseVal().numberOfItems() > 0 )
            {
                posY = static_cast<double> ( y().baseVal().getItem ( 0 ).value() );
            }

            // Render direct text children.  For tspan children the
            // recursive DrawStart call will handle them.
            for ( const auto& child : childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child );
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
