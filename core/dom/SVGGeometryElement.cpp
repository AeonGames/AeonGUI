/*
Copyright (C) 2020,2024 Rodrigo Jose Hernandez Cordoba

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
#include "SVGGeometryElement.h"
#include "CSSSelectHandler.h"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        SVGGeometryElement::SVGGeometryElement ( const std::string& aTagName, const AttributeMap& aAttributes ) : SVGGraphicsElement ( aTagName, aAttributes ), mPath{}
        {
        }
        SVGGeometryElement::~SVGGeometryElement() = default;
        void SVGGeometryElement::DrawStart ( Canvas& aCanvas ) const
        {
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_color color{};
                css_fixed opacity{};
                float fopacity{};
                uint8_t type{ css_computed_fill ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &color ) };
                std::cout << "Fill: " << std::hex << color << " Type: " << static_cast<uint32_t> ( type ) << std::endl;
                aCanvas.SetFillColor ( Color{color} );
                type = css_computed_stroke ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &color );
                std::cout << "Stroke: " << std::hex << color << " Type: " << static_cast<uint32_t> ( type ) << std::endl;
                aCanvas.SetStrokeColor ( Color{color} );
                type = css_computed_fill_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &opacity );
                fopacity = FIXTOFLT ( opacity );
                std::cout << "Fill Opacity: " << fopacity << " Type: " << static_cast<uint32_t> ( type ) << std::endl;
                aCanvas.SetFillOpacity ( FIXTOFLT ( opacity ) );
                type = css_computed_stroke_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &opacity );
                fopacity = FIXTOFLT ( opacity );
                std::cout << "Stroke Opacity: " << fopacity << " Type: " << static_cast<uint32_t> ( type ) << std::endl;
                aCanvas.SetStrokeOpacity ( FIXTOFLT ( opacity ) );
                fopacity = FIXTOFLT ( opacity );
                type = css_computed_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &opacity );
                std::cout << "Opacity: " << fopacity << " Type: " << static_cast<uint32_t> ( type ) << std::endl;
                aCanvas.SetOpacity ( FIXTOFLT ( opacity ) );
            }
#if 0
            // TODO: convert to use libcss
            aCanvas.SetFillColor ( std::get<ColorAttr> ( GetInheritedAttribute ( "fill", Color{black} ) ) );
            aCanvas.SetStrokeColor ( std::get<ColorAttr> ( GetInheritedAttribute ( "stroke", ColorAttr{} ) ) );
            aCanvas.SetStrokeWidth ( std::get<double> ( GetInheritedAttribute ( "stroke-width", 1.0 ) ) );
            aCanvas.SetStrokeOpacity ( std::get<double> ( GetInheritedAttribute ( "stroke-opacity", 1.0 ) ) );
            aCanvas.SetFillOpacity ( std::get<double> ( GetInheritedAttribute ( "fill-opacity", 1.0 ) ) );
            aCanvas.SetOpacity ( std::get<double> ( GetInheritedAttribute ( "opacity", 1.0 ) ) );
#endif
            aCanvas.Draw ( mPath );
        }
    }
}
