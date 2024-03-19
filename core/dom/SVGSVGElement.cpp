/*
Copyright (C) 2019,2020,2023,2024 Rodrigo Jose Hernandez Cordoba

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
#include "SVGSVGElement.h"

namespace AeonGUI
{
    namespace DOM
    {
        SVGSVGElement::SVGSVGElement ( const std::string& aTagName, const AttributeMap& aAttributes ) : SVGGraphicsElement { aTagName, aAttributes }
        {
            std::cout << "This is a specialized implementation for the svg element." << std::endl;
            if ( aAttributes.find ( "viewBox" ) != aAttributes.end() )
            {
                static const std::regex viewBoxRegex{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};

                const std::string& viewBoxStr{aAttributes.at ( "viewBox" ) };
                std::smatch match{};
                if ( std::regex_match ( viewBoxStr, match, viewBoxRegex ) )
                {
                    mViewBox.mWidth  = std::stod ( match[1] );
                    mViewBox.mHeight = std::stod ( match[2] );
                    mViewBox.mX      = std::stod ( match[3] );
                    mViewBox.mY      = std::stod ( match[4] );
                }
            }
            if ( aAttributes.find ( "width" ) != aAttributes.end() )
            {
                mWidth = std::stod ( aAttributes.at ( "width" ) );
            }
            if ( aAttributes.find ( "height" ) != aAttributes.end() )
            {
                mHeight = std::stod ( aAttributes.at ( "height" ) );
            }
        }
        SVGSVGElement::~SVGSVGElement()
        {
        }
        void SVGSVGElement::DrawStart ( Canvas& aCanvas ) const
        {
        }
    }
}
