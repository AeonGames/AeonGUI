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
#include "SVGLineElement.h"
#include <iostream>

namespace AeonGUI
{
    namespace Elements
    {
        SVGLineElement::SVGLineElement ( xmlElementPtr aXmlElementPtr ) : SVGGeometryElement ( aXmlElementPtr )
        {
            std::cout << "Line" << std::endl;
            /**
             * https://www.w3.org/TR/SVG/shapes.html#LineElement
            */
            double x1 = std::get<double> ( GetAttribute ( "x1", 0.0 ) );
            double y1 = std::get<double> ( GetAttribute ( "y1", 0.0 ) );
            double x2 = std::get<double> ( GetAttribute ( "x2", 0.0 ) );
            double y2 = std::get<double> ( GetAttribute ( "y2", 0.0 ) );
            std::vector<DrawType> path
            {
                /// 1. perform an absolute moveto operation to absolute location (x1,y1)
                static_cast<uint64_t> ( 'M' ), x1, y1,
                /// 2. perform an absolute lineto operation to absolute location (x2,y2)
                static_cast<uint64_t> ( 'L' ), x2, y2,
            };
            mPath.Construct ( path );
        }
        SVGLineElement::~SVGLineElement()
        {
        }
    }
}
