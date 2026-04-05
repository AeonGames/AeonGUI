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
#include "aeongui/dom/SVGLength.hpp"
#include "aeongui/Canvas.hpp"
#include <iostream>

namespace AeonGUI
{
    namespace DOM
    {
        bool SVGLineElement::IsPercentage ( const std::string& aStr )
        {
            size_t pos{};
            std::stod ( aStr, &pos );
            return pos < aStr.size() && aStr[pos] == '%';
        }

        SVGLineElement::SVGLineElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, std::move ( aAttributes ), aParent }
        {
            std::cout << "Line" << std::endl;
            if ( mAttributes.find ( "x1" ) != mAttributes.end() )
            {
                const auto& val = mAttributes.at ( "x1" );
                mX1Pct = IsPercentage ( val );
                mX1Raw = std::stod ( val );
                mX1 = mX1Raw;
            }
            if ( mAttributes.find ( "y1" ) != mAttributes.end() )
            {
                const auto& val = mAttributes.at ( "y1" );
                mY1Pct = IsPercentage ( val );
                mY1Raw = std::stod ( val );
                mY1 = mY1Raw;
            }
            if ( mAttributes.find ( "x2" ) != mAttributes.end() )
            {
                const auto& val = mAttributes.at ( "x2" );
                mX2Pct = IsPercentage ( val );
                mX2Raw = std::stod ( val );
                mX2 = mX2Raw;
            }
            if ( mAttributes.find ( "y2" ) != mAttributes.end() )
            {
                const auto& val = mAttributes.at ( "y2" );
                mY2Pct = IsPercentage ( val );
                mY2Raw = std::stod ( val );
                mY2 = mY2Raw;
            }
            if ( !mX1Pct && !mY1Pct && !mX2Pct && !mY2Pct )
            {
                BuildPath();
            }
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
                mX1Pct = IsPercentage ( aValue );
                mX1Raw = std::stod ( aValue );
                mX1 = mX1Raw;
                mLastVpWidth = -1;
            }
            else if ( aName == "y1" )
            {
                mY1Pct = IsPercentage ( aValue );
                mY1Raw = std::stod ( aValue );
                mY1 = mY1Raw;
                mLastVpHeight = -1;
            }
            else if ( aName == "x2" )
            {
                mX2Pct = IsPercentage ( aValue );
                mX2Raw = std::stod ( aValue );
                mX2 = mX2Raw;
                mLastVpWidth = -1;
            }
            else if ( aName == "y2" )
            {
                mY2Pct = IsPercentage ( aValue );
                mY2Raw = std::stod ( aValue );
                mY2 = mY2Raw;
                mLastVpHeight = -1;
            }
            else
            {
                return;
            }
            if ( !mX1Pct && !mY1Pct && !mX2Pct && !mY2Pct )
            {
                BuildPath();
            }
        }

        void SVGLineElement::ResolveViewportPercentages ( const Canvas& aCanvas ) const
        {
            if ( !mX1Pct && !mY1Pct && !mX2Pct && !mY2Pct )
            {
                return;
            }
            double vpW = aCanvas.GetViewportWidth();
            double vpH = aCanvas.GetViewportHeight();
            if ( vpW == mLastVpWidth && vpH == mLastVpHeight )
            {
                return;
            }
            mLastVpWidth  = vpW;
            mLastVpHeight = vpH;
            if ( mX1Pct )
            {
                mX1 = mX1Raw * vpW / 100.0;
            }
            if ( mY1Pct )
            {
                mY1 = mY1Raw * vpH / 100.0;
            }
            if ( mX2Pct )
            {
                mX2 = mX2Raw * vpW / 100.0;
            }
            if ( mY2Pct )
            {
                mY2 = mY2Raw * vpH / 100.0;
            }
            const_cast<SVGLineElement * > ( this )->BuildPath();
        }
    }
}
