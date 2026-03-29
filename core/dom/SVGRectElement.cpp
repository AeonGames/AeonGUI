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
#include <iostream>
#include <array>
#include "aeongui/dom/SVGRectElement.hpp"
#include "aeongui/dom/SVGAnimateElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGRectElement::SVGRectElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement {aTagName, std::move ( aAttributes ), aParent}
        {
            std::cout << "Rect" << std::endl;
            if ( mAttributes.find ( "width" ) != mAttributes.end() )
            {
                mWidth = std::stod ( mAttributes.at ( "width" ) );
            }
            if ( mAttributes.find ( "height" ) != mAttributes.end() )
            {
                mHeight = std::stod ( mAttributes.at ( "height" ) );
            }
            if ( mAttributes.find ( "x" ) != mAttributes.end() )
            {
                mX = std::stod ( mAttributes.at ( "x" ) );
            }
            if ( mAttributes.find ( "y" ) != mAttributes.end() )
            {
                mY = std::stod ( mAttributes.at ( "y" ) );
            }
            if ( mAttributes.find ( "rx" ) != mAttributes.end() )
            {
                mRx = std::stod ( mAttributes.at ( "rx" ) );
            }
            if ( mAttributes.find ( "ry" ) != mAttributes.end() )
            {
                mRy = std::stod ( mAttributes.at ( "ry" ) );
            }
            // SVG spec: if only rx or ry is specified, the other defaults to the same value.
            if ( mRx > 0.0 && mRy == 0.0 )
            {
                mRy = mRx;
            }
            else if ( mRy > 0.0 && mRx == 0.0 )
            {
                mRx = mRy;
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

        void SVGRectElement::RebuildAnimatedPath() const
        {
            double rx = mRx;
            double ry = mRy;
            bool needsRebuild = false;
            for ( const auto& child : childNodes() )
            {
                if ( auto * anim = dynamic_cast<const SVGAnimateElement * > ( child.get() ) )
                {
                    if ( anim->IsActive() && anim->IsPathAnimation() )
                    {
                        const auto& name = anim->GetAttributeName();
                        if ( name == "rx" )
                        {
                            rx = anim->GetInterpolatedValue();
                            needsRebuild = true;
                        }
                        else if ( name == "ry" )
                        {
                            ry = anim->GetInterpolatedValue();
                            needsRebuild = true;
                        }
                    }
                }
            }
            if ( !needsRebuild || mWidth <= 0.0 || mHeight <= 0.0 )
            {
                return;
            }
            std::array<DrawType, 44> path{};
            size_t i = 0;
            path[i++] = static_cast<uint64_t> ( 'M' );
            path[i++] = rx + mX;
            path[i++] = mY;
            path[i++] = static_cast<uint64_t> ( 'H' );
            path[i++] = mX + mWidth - rx;
            if ( rx > 0.0 && ry > 0.0 )
            {
                path[i++] = static_cast<uint64_t> ( 'A' );
                path[i++] = rx;
                path[i++] = ry;
                path[i++] = 0.0;
                path[i++] = false;
                path[i++] = true;
                path[i++] = mX + mWidth;
                path[i++] = mY + ry;
            }
            path[i++] = static_cast<uint64_t> ( 'V' );
            path[i++] = mY + mHeight - ry;
            if ( rx > 0.0 && ry > 0.0 )
            {
                path[i++] = static_cast<uint64_t> ( 'A' );
                path[i++] = rx;
                path[i++] = ry;
                path[i++] = 0.0;
                path[i++] = false;
                path[i++] = true;
                path[i++] = mX + mWidth - rx;
                path[i++] = mY + mHeight;
            }
            path[i++] = static_cast<uint64_t> ( 'H' );
            path[i++] = mX + rx;
            if ( rx > 0.0 && ry > 0.0 )
            {
                path[i++] = static_cast<uint64_t> ( 'A' );
                path[i++] = rx;
                path[i++] = ry;
                path[i++] = 0.0;
                path[i++] = false;
                path[i++] = true;
                path[i++] = mX;
                path[i++] = mY + mHeight - ry;
            }
            path[i++] = static_cast<uint64_t> ( 'V' );
            path[i++] = mY + ry;
            if ( rx > 0.0 && ry > 0.0 )
            {
                path[i++] = static_cast<uint64_t> ( 'A' );
                path[i++] = rx;
                path[i++] = ry;
                path[i++] = 0.0;
                path[i++] = false;
                path[i++] = true;
                path[i++] = rx + mX;
                path[i++] = mY;
            }
            path[i++] = static_cast<uint64_t> ( 'Z' );
            mPath.Construct ( path.data(), i );
        }
    }
}
