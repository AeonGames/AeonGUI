/*
Copyright (C) 2019,2020,2023-2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include <regex>
#include "aeongui/Canvas.hpp"
#include "aeongui/dom/SVGSVGElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGSVGElement::SVGSVGElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) :
            SVGGraphicsElement { aTagName, std::move ( aAttributes ), aParent }
        {
            std::cout << "This is a specialized implementation for the svg element." << std::endl;
            if ( mAttributes.find ( "viewBox" ) != mAttributes.end() )
            {
                static const std::regex viewBoxRegex{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};

                const std::string& viewBoxStr{mAttributes.at ( "viewBox" ) };
                std::smatch match{};
                if ( std::regex_match ( viewBoxStr, match, viewBoxRegex ) )
                {
                    mViewBox.width  = std::stod ( match[3] );
                    mViewBox.height = std::stod ( match[4] );
                    mViewBox.min_x      = std::stod ( match[1] );
                    mViewBox.min_y      = std::stod ( match[2] );
                }
            }
            if ( mAttributes.find ( "width" ) != mAttributes.end() )
            {
                mWidth = std::stod ( mAttributes.at ( "width" ) );
            }
            if ( mAttributes.find ( "height" ) != mAttributes.end() )
            {
                mHeight = std::stod ( mAttributes.at ( "height" ) );
            }
            if ( mAttributes.find ( "preserveAspectRatio" ) != mAttributes.end() )
            {
                mPreserveAspectRatio = PreserveAspectRatio{mAttributes.at ( "preserveAspectRatio" ) };
            }
        }

        SVGSVGElement::~SVGSVGElement() = default;

        void SVGSVGElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );
            aCanvas.SetViewBox ( mViewBox, mPreserveAspectRatio );
        }
    }
}
