/*
Copyright (C) 2019,2020,2024,2025 Rodrigo Jose Hernandez Cordoba

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
#include <array>
#include "aeongui/dom/SVGRectElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGRectElement::SVGRectElement ( const std::string& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGGeometryElement {aTagName, aAttributes, aParent}
        {
            std::cout << "Rect" << std::endl;
            if ( aAttributes.find ( "width" ) != aAttributes.end() )
            {
                mWidth = std::stod ( aAttributes.at ( "width" ) );
            }
            if ( aAttributes.find ( "height" ) != aAttributes.end() )
            {
                mHeight = std::stod ( aAttributes.at ( "height" ) );
            }
            if ( aAttributes.find ( "x" ) != aAttributes.end() )
            {
                mX = std::stod ( aAttributes.at ( "x" ) );
            }
            if ( aAttributes.find ( "y" ) != aAttributes.end() )
            {
                mY = std::stod ( aAttributes.at ( "y" ) );
            }
            if ( aAttributes.find ( "rx" ) != aAttributes.end() )
            {
                mRx = std::stod ( aAttributes.at ( "rx" ) );
            }
            if ( aAttributes.find ( "ry" ) != aAttributes.end() )
            {
                mRy = std::stod ( aAttributes.at ( "ry" ) );
            }
            /**
             * https://www.w3.org/TR/SVG/shapes.html#RectElement
             * The width and height properties define the overall width and height of the rectangle.
             * A negative value for either property is illegal and must be ignored as a parsing error.
             * A computed value of zero for either dimension disables rendering of the element.
            */
            if ( ( mWidth > 0.0 ) && ( mHeight > 0.0 ) )
            {
                std::array<DrawType, 44> path{};
                size_t i = 0;
                /// 1. perform an absolute moveto operation to location (x+rx,y);
                path[i++] = static_cast<uint64_t> ( 'M' );
                path[i++] = mRx + mX;
                path[i++] = mY;
                /// 2. perform an absolute horizontal lineto with parameter x+width-rx;
                path[i++] = static_cast<uint64_t> ( 'H' );
                path[i++] = mX + mWidth - mRx;
                /// 3. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x+width,y+ry), where rx and ry are used as the equivalent parameters to the elliptical arc command, the x-axis-rotation and large-arc-flag are set to zero, the sweep-flag is set to one;
                if ( mRx > 0.0 && mRy > 0.0 )
                {
                    path[i++] = static_cast<uint64_t> ( 'A' );
                    path[i++] = mRx;
                    path[i++] = mRy;
                    path[i++] = 0.0;
                    path[i++] = false;
                    path[i++] = true;
                    path[i++] = mX + mWidth;
                    path[i++] = mY + mRy;
                }
                /// 4. perform an absolute vertical lineto parameter y+height-ry;
                path[i++] = static_cast<uint64_t> ( 'V' );
                path[i++] = mY + mHeight - mRy;
                /// 5. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x+width-rx,y+height), using the same parameters as previously;
                if ( mRx > 0.0 && mRy > 0.0 )
                {
                    path[i++] = static_cast<uint64_t> ( 'A' );
                    path[i++] = mRx;
                    path[i++] = mRy;
                    path[i++] = 0.0;
                    path[i++] = false;
                    path[i++] = true;
                    path[i++] = mX + mWidth - mRx;
                    path[i++] = mY + mHeight;
                }
                /// 6. perform an absolute horizontal lineto parameter x+rx;
                path[i++] = static_cast<uint64_t> ( 'H' );
                path[i++] = mX + mRx;
                /// 7. if both rx and ry are greater than zero, perform an absolute elliptical arc operation to coordinate (x,y+height-ry), using the same parameters as previously;
                if ( mRx > 0.0 && mRy > 0.0 )
                {
                    path[i++] = static_cast<uint64_t> ( 'A' );
                    path[i++] = mRx;
                    path[i++] = mRy;
                    path[i++] = 0.0;
                    path[i++] = false;
                    path[i++] = true;
                    path[i++] = mX;
                    path[i++] = mY + mHeight - mRy;
                }
                /// 8. perform an absolute vertical lineto parameter y+ry
                path[i++] = static_cast<uint64_t> ( 'V' );
                path[i++] = mY + mRy;
                /// 9. if both rx and ry are greater than zero, perform an absolute elliptical arc operation with a segment-completing close path operation, using the same parameters as previously.
                if ( mRx > 0.0 && mRy > 0.0 )
                {
                    path[i++] = static_cast<uint64_t> ( 'A' );
                    path[i++] = mRx;
                    path[i++] = mRy;
                    path[i++] = 0.0;
                    path[i++] = false;
                    path[i++] = true;
                    path[i++] = mRx + mX;
                    path[i++] = mY;
                }
                // 10. close path.
                path[i++] = static_cast<uint64_t> ( 'Z' );
                std::cout << i << std::endl;
                mPath.Construct ( path.data(), i );
            }
        }

        SVGRectElement::~SVGRectElement() = default;
    }
}
