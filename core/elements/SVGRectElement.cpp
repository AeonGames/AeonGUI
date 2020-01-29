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
#include <vector>
#include "SVGRectElement.h"

namespace AeonGUI
{
    namespace Elements
    {
        SVGRectElement::SVGRectElement ( xmlElementPtr aXmlElementPtr ) : SVGGeometryElement ( aXmlElementPtr )
        {
            std::cout << "Rect" << std::endl;
            double width = GetAttrAsDouble ( "width" );
            double height = GetAttrAsDouble ( "height" );
            /**
             * https://www.w3.org/TR/SVG/shapes.html#RectElement
             * The width and height properties define the overall width and height of the rectangle.
             * A negative value for either property is illegal and must be ignored as a parsing error.
             * A computed value of zero for either dimension disables rendering of the element.
            */
            if ( ( width > 0.0 ) && ( height > 0.0 ) )
            {
                double x = GetAttrAsDouble ( "x" );
                double y = GetAttrAsDouble ( "y" );
                double rx = GetAttrAsDouble ( "rx" );
                double ry = GetAttrAsDouble ( "ry" );
                std::vector<DrawType> path
                {
                    /// 1. perform an absolute moveto operation to location (x+rx,y);
                    static_cast<uint64_t> ( 'M' ), rx + x, y,
                    /// 2. perform an absolute horizontal lineto with parameter x+width-rx;
                    static_cast<uint64_t> ( 'H' ), x + width - rx,
                    /// 3. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x+width,y+ry), where rx and ry are used as the equivalent parameters to the elliptical arc command, the x-axis-rotation and large-arc-flag are set to zero, the sweep-flag is set to one;
                    /// 4. perform an absolute vertical lineto parameter y+height-ry;
                    static_cast<uint64_t> ( 'V' ), y + height - ry,
                    /// 5. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x+width-rx,y+height), using the same parameters as previously;
                    /// 6. perform an absolute horizontal lineto parameter x+rx;
                    static_cast<uint64_t> ( 'H' ), x + rx,
                    /// 7. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x,y+height-ry), using the same parameters as previously;
                    /// 8. perform an absolute vertical lineto parameter y+ry
                    static_cast<uint64_t> ( 'V' ), y + ry,
                    /// 9. if both rx and ry are greater than zero, perform an absolute elliptical arc operation with a segment-completing close path operation, using the same parameters as previously.
                    // 10. close path.
                    static_cast<uint64_t> ( 'Z' ),
                };
                mPath.Construct ( path );
            }
        }

        SVGRectElement::~SVGRectElement() = default;
    }
}
