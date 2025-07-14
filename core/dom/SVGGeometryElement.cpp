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
#include <iostream>
#include "aeongui/dom/SVGGeometryElement.h"
#include "aeongui/dom/CSSSelectHandler.h"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        SVGGeometryElement::SVGGeometryElement ( const std::string& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGGraphicsElement ( aTagName, aAttributes, aParent ), mPath{}
        {
        }
        SVGGeometryElement::~SVGGeometryElement() = default;
        void SVGGeometryElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_color color{};
                css_fixed fixed{};
                css_unit unit{};
                if ( css_computed_fill ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &color ) != CSS_PAINT_NONE )
                {
                    aCanvas.SetFillColor ( Color{color} );
                    css_computed_fill_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &fixed );
                    aCanvas.SetFillOpacity ( FIXTOFLT ( fixed ) );
                }
                if ( css_computed_stroke ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &color ) )
                {
                    aCanvas.SetStrokeColor ( Color{color} );
                    css_computed_stroke_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &fixed );
                    aCanvas.SetStrokeOpacity ( FIXTOFLT ( fixed ) );
                    css_computed_stroke_width ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &fixed, &unit );
                    /// @todo Convert unit to pixels if necessary
                    aCanvas.SetStrokeWidth ( FIXTOFLT ( fixed ) );
                }
                css_computed_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &fixed );
                aCanvas.SetOpacity ( FIXTOFLT ( fixed ) );
            }
            aCanvas.Draw ( mPath );
        }
    }
}
