/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGSetElement.hpp"
#include "aeongui/Canvas.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGSetElement::SVGSetElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGAnimationElement { aTagName, std::move ( aAttributes ), aParent }
        {
            if ( mAttributes.find ( "to" ) != mAttributes.end() )
            {
                mToValue = mAttributes.at ( "to" );
            }
            mIsColorAttribute = ( mAttributeName == "fill" || mAttributeName == "stroke" );
            if ( mIsColorAttribute )
            {
                mColorValue = Color { mToValue };
            }
        }

        SVGSetElement::~SVGSetElement() = default;

        void SVGSetElement::ApplyToCanvas ( Canvas& aCanvas ) const
        {
            if ( !mIsActive )
            {
                return;
            }
            if ( mIsColorAttribute )
            {
                if ( mAttributeName == "fill" )
                {
                    aCanvas.SetFillColor ( mColorValue );
                }
                else if ( mAttributeName == "stroke" )
                {
                    aCanvas.SetStrokeColor ( mColorValue );
                }
            }
            else
            {
                double value = std::stod ( mToValue );
                if ( mAttributeName == "opacity" )
                {
                    aCanvas.SetOpacity ( value );
                }
                else if ( mAttributeName == "stroke-width" )
                {
                    aCanvas.SetStrokeWidth ( value );
                }
            }
        }
    }
}
