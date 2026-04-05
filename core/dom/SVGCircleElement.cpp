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
#include <cmath>
#include "aeongui/dom/SVGCircleElement.hpp"
#include "aeongui/dom/SVGLength.hpp"
#include "aeongui/Canvas.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        static bool HasPercent ( const AttributeMap& attrs, const char* name )
        {
            auto it = attrs.find ( name );
            if ( it == attrs.end() )
            {
                return false;
            }
            return it->second.find ( '%' ) != std::string::npos;
        }

        SVGCircleElement::SVGCircleElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, std::move ( aAttributes ), aParent }
        {
            std::cout << "Circle" << std::endl;
            mHasPercentage = HasPercent ( mAttributes, "cx" ) || HasPercent ( mAttributes, "cy" ) || HasPercent ( mAttributes, "r" );
            if ( !mHasPercentage )
            {
                BuildPath ( 0, 0 );
            }
        }

        SVGCircleElement::~SVGCircleElement()
        {
        }

        void SVGCircleElement::BuildPath ( double aVpW, double aVpH )
        {
            auto parse = [&] ( const char* name, double ref ) -> double
            {
                auto it = mAttributes.find ( name );
                return it != mAttributes.end() ? SVGLength::ParseAttribute ( it->second, ref ) : 0.0;
            };
            double diagonal = std::sqrt ( aVpW * aVpW + aVpH * aVpH ) / std::sqrt ( 2.0 );
            double cx = parse ( "cx", aVpW );
            double cy = parse ( "cy", aVpH );
            double r  = parse ( "r",  diagonal );
            if ( r > 0.0 )
            {
                std::vector<DrawType> path
                {
                    static_cast<uint64_t> ( 'M' ), cx + r, cy,
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx, cy + r,
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx - r, cy,
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx, cy - r,
                    static_cast<uint64_t> ( 'A' ), r, r, 0.0, false, true, cx + r, cy,
                    static_cast<uint64_t> ( 'Z' ),
                };
                mPath->Construct ( path );
            }
        }

        void SVGCircleElement::ResolveViewportPercentages ( const Canvas& aCanvas ) const
        {
            if ( !mHasPercentage )
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
            const_cast<SVGCircleElement*> ( this )->BuildPath ( vpW, vpH );
        }

        void SVGCircleElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "cx" || aName == "cy" || aName == "r" )
            {
                mHasPercentage = HasPercent ( mAttributes, "cx" ) || HasPercent ( mAttributes, "cy" ) || HasPercent ( mAttributes, "r" );
                mLastVpWidth = -1;
                if ( !mHasPercentage )
                {
                    BuildPath ( 0, 0 );
                }
            }
        }
    }
}
