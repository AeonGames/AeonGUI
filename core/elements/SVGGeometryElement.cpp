/*
Copyright (C) 2020 Rodrigo Jose Hernandez Cordoba

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

namespace AeonGUI
{
    namespace Elements
    {
        SVGGeometryElement::SVGGeometryElement ( xmlElementPtr aXmlElementPtr ) : SVGGraphicsElement ( aXmlElementPtr ), mPath{}
        {
        }
        SVGGeometryElement::~SVGGeometryElement() = default;
        void SVGGeometryElement::DrawStart ( Canvas& aCanvas ) const
        {
            /** @todo Line based paths use different defaults than the ones listed here. */
            aCanvas.SetFillColor ( std::get<ColorAttr> ( GetInheritedAttribute ( "fill", Color{black} ) ) );
            aCanvas.SetStrokeColor ( std::get<ColorAttr> ( GetInheritedAttribute ( "stroke", ColorAttr{} ) ) );
            aCanvas.SetStrokeWidth ( std::get<double> ( GetInheritedAttribute ( "stroke-width", 1.0 ) ) );
            aCanvas.SetStrokeOpacity ( std::get<double> ( GetInheritedAttribute ( "stroke-opacity", 1.0 ) ) );
            aCanvas.SetFillOpacity ( std::get<double> ( GetInheritedAttribute ( "fill-opacity", 1.0 ) ) );
            aCanvas.SetOpacity ( std::get<double> ( GetInheritedAttribute ( "opacity", 1.0 ) ) );
            aCanvas.Draw ( mPath );
        }
    }
}
