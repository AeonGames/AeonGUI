/*
Copyright (C) 2019,2020,2024 Rodrigo Jose Hernandez Cordoba

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
    namespace DOM
    {
        SVGLineElement::SVGLineElement ( const std::string& aTagName, const AttributeMap& aAttributes ) : SVGGeometryElement { aTagName, aAttributes }
        {
            std::cout << "Line" << std::endl;
            /**
             * https://www.w3.org/TR/SVG/shapes.html#LineElement
            */
            if ( aAttributes.find ( "x1" ) != aAttributes.end() )
            {
                mX1 = std::stod ( aAttributes.at ( "x1" ) );
            }
            if ( aAttributes.find ( "y1" ) != aAttributes.end() )
            {
                mY1 = std::stod ( aAttributes.at ( "y1" ) );
            }
            if ( aAttributes.find ( "x2" ) != aAttributes.end() )
            {
                mX2 = std::stod ( aAttributes.at ( "x2" ) );
            }
            if ( aAttributes.find ( "y2" ) != aAttributes.end() )
            {
                mY2 = std::stod ( aAttributes.at ( "y2" ) );
            }
            std::vector<DrawType> path
            {
                /// 1. perform an absolute moveto operation to absolute location (x1,y1)
                static_cast<uint64_t> ( 'M' ), mX1, mY1,
                /// 2. perform an absolute lineto operation to absolute location (x2,y2)
                static_cast<uint64_t> ( 'L' ), mX2, mY2,
            };
            mPath.Construct ( path );
        }
        SVGLineElement::~SVGLineElement()
        {
        }
    }
}
