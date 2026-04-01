/*
Copyright (C) 2019,2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGLineElement.hpp"
#include <iostream>

namespace AeonGUI
{
    namespace DOM
    {
        SVGLineElement::SVGLineElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, std::move ( aAttributes ), aParent }
        {
            std::cout << "Line" << std::endl;
            /**
             * https://www.w3.org/TR/SVG/shapes.html#LineElement
            */
            if ( mAttributes.find ( "x1" ) != mAttributes.end() )
            {
                mX1 = std::stod ( mAttributes.at ( "x1" ) );
            }
            if ( mAttributes.find ( "y1" ) != mAttributes.end() )
            {
                mY1 = std::stod ( mAttributes.at ( "y1" ) );
            }
            if ( mAttributes.find ( "x2" ) != mAttributes.end() )
            {
                mX2 = std::stod ( mAttributes.at ( "x2" ) );
            }
            if ( mAttributes.find ( "y2" ) != mAttributes.end() )
            {
                mY2 = std::stod ( mAttributes.at ( "y2" ) );
            }
            BuildPath();
        }

        SVGLineElement::~SVGLineElement()
        {
        }

        void SVGLineElement::BuildPath()
        {
            std::vector<DrawType> path
            {
                /// 1. perform an absolute moveto operation to absolute location (x1,y1)
                static_cast<uint64_t> ( 'M' ), mX1, mY1,
                /// 2. perform an absolute lineto operation to absolute location (x2,y2)
                static_cast<uint64_t> ( 'L' ), mX2, mY2,
            };
            mPath->Construct ( path );
        }

        void SVGLineElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "x1" )
            {
                mX1 = std::stod ( aValue );
            }
            else if ( aName == "y1" )
            {
                mY1 = std::stod ( aValue );
            }
            else if ( aName == "x2" )
            {
                mX2 = std::stod ( aValue );
            }
            else if ( aName == "y2" )
            {
                mY2 = std::stod ( aValue );
            }
            else
            {
                return;
            }
            BuildPath();
        }
    }
}
