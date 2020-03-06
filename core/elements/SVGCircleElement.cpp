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
#include "SVGCircleElement.h"

namespace AeonGUI
{
    namespace Elements
    {
        SVGCircleElement::SVGCircleElement ( xmlElementPtr aXmlElementPtr ) : SVGGeometryElement ( aXmlElementPtr )
        {
            std::cout << "Circle" << std::endl;
            double cx = std::get<double> ( GetAttribute ( "cx", 0.0 ) );
            double cy = std::get<double> ( GetAttribute ( "cy", 0.0 ) );
            double r = std::get<double> ( GetAttribute ( "r", 0.0 ) );
            /**
             * https://www.w3.org/TR/SVG/shapes.html#CircleElement
             * The cx and cy attributes define the coordinates of the center of the circle.
             * The r attribute defines the radius of the circle. A negative value for either property is illegal and must be ignored as a parsing error.
            */
            if ( r > 0.0 )
            {
                std::vector<DrawType> path
                {
                    // 1. A move-to command to the point cx+r,cy;
                    static_cast<uint64_t> ( 'M' ), cx + r, cy,
                    // 2. arc to cx,cy+r;
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx, cy + r,
                    // 3. arc to cx-r,cy;
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx - r, cy,
                    // 4. arc to cx,cy-r;
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx, cy - r,
                    // 5. arc with a segment-completing close path operation.
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx + r, cy,
                    // 6. close path.
                    static_cast<uint64_t> ( 'Z' ),
                };
                mPath.Construct ( path );
            }
        }
        SVGCircleElement::~SVGCircleElement()
        {
        }
    }
}
