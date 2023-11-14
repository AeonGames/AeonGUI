/*
Copyright (C) 2019,2020,2023 Rodrigo Jose Hernandez Cordoba

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
            auto viewBox{ GetAttribute ( "viewBox" ) };
            auto width{ GetAttribute ( "width" ) };
            auto height{ GetAttribute ( "height" ) };
            if ( std::holds_alternative<std::string> ( viewBox ) )
            {
                static const std::regex viewBoxRegex{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};

                const std::string& viewBoxStr{std::get<std::string> ( viewBox ) };
                std::smatch match{};
                if ( std::regex_match ( viewBoxStr, match, viewBoxRegex ) )
                {
                    mViewBox.mWidth  = std::stod ( match[1] );
                    mViewBox.mHeight = std::stod ( match[2] );
                    mViewBox.mX      = std::stod ( match[3] );
                    mViewBox.mY      = std::stod ( match[4] );
                }
            }
            if ( std::holds_alternative<double> ( width ) )
            {
                mViewPort.mWidth = std::get<double> ( width );
            }
            if ( std::holds_alternative<double> ( height ) )
            {
                mViewPort.mHeight = std::get<double> ( height );
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
