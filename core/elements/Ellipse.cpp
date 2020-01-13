/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include "Ellipse.h"

namespace AeonGUI
{
    namespace Elements
    {

        Ellipse::Ellipse ( xmlElementPtr aXmlElementPtr ) : Element ( aXmlElementPtr )
        {
            std::cout << "Ellipse" << std::endl;
            double cx = GetAttrAsDouble ( "cx" );
            double cy = GetAttrAsDouble ( "cy" );
            double rx = GetAttrAsDouble ( "rx" );
            double ry = GetAttrAsDouble ( "ry" );
            /**
             * https://www.w3.org/TR/SVG/shapes.html#EllipseElement
             * The cx and cy coordinates define the center of the ellipse.
             * The rx and ry properties define the x- and y-axis radii of the ellipse.
             * A negative value for either property is illegal and must be ignored as a parsing error.
             * A computed value of zero for either dimension, or a computed value of auto for both dimensions, disables rendering of the element.
            */
            if ( rx > 0.0 && ry > 0.0 )
            {
                std::vector<DrawType> path
                {
                    // 1. A move-to command to the point cx+rx,cy;
                    static_cast<uint64_t> ( 'M' ), cx + rx, cy,
                    // 2. arc to cx,cy+ry;
                    static_cast<uint64_t> ( 'A' ), rx, ry, 0.0, false, true, cx, cy + ry,
                    // 3. arc to cx-rx,cy;
                    static_cast<uint64_t> ( 'A' ), rx, ry, 0.0, false, true, cx - rx, cy,
                    // 4. arc to cx,cy-ry;
                    static_cast<uint64_t> ( 'A' ), rx, ry, 0.0, false, true, cx, cy - ry,
                    // 5. arc with a segment-completing close path operation.
                    static_cast<uint64_t> ( 'A' ), rx, ry, 0.0, false, true, cx + rx, cy,
                    // 6. close path.
                    static_cast<uint64_t> ( 'Z' ),
                };
                mPath.Construct ( path );
            }
        }
        Ellipse::~Ellipse()
        {
        }
        void Ellipse::DrawStart ( Canvas& aCanvas ) const
        {
            // Default Workaround
            aCanvas.SetFillColor ( black );
            /// @todo All these should probably have explicit defaults set.
            for ( auto& i : mAttributeMap )
            {
                if ( i.first == "fill" )
                {
                    aCanvas.SetFillColor ( std::get<Color> ( i.second ) );
                }
                else if ( i.first == "stroke" )
                {
                    aCanvas.SetStrokeColor ( std::get<Color> ( i.second ) );
                }
                else if ( i.first == "stroke-width" )
                {
                    aCanvas.SetStrokeWidth ( std::get<double> ( i.second ) );
                }
                else if ( i.first == "stroke-opacity" )
                {
                    aCanvas.SetStrokeOpacity ( std::get<double> ( i.second ) );
                }
                else if ( i.first == "fill-opacity" )
                {
                    aCanvas.SetFillOpacity ( std::get<double> ( i.second ) );
                }
            }
            aCanvas.Draw ( mPath );
        }
    }
}