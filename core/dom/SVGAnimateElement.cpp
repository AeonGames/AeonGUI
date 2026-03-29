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
#include <algorithm>
#include <cmath>
#include "aeongui/dom/SVGAnimateElement.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/Matrix2x3.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGAnimateElement::SVGAnimateElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGAnimationElement { aTagName, std::move ( aAttributes ), aParent }
        {
            // Determine if this is a color or numeric animation
            mIsColorAnimation = ( mAttributeName == "fill" || mAttributeName == "stroke" );

            // Determine if this is a geometry attribute animation
            mIsGeometryAnimation = ( mAttributeName == "cx" || mAttributeName == "cy" ||
                                     mAttributeName == "x" || mAttributeName == "y" ||
                                     mAttributeName == "width" || mAttributeName == "height" ||
                                     mAttributeName == "r" );

            // Determine if this is a path-modifying animation (requires path rebuild)
            mIsPathAnimation = ( mAttributeName == "rx" || mAttributeName == "ry" );

            // Read the parent's original attribute value for geometry animations
            if ( mIsGeometryAnimation && aParent && aParent->nodeType() == Node::ELEMENT_NODE )
            {
                auto* parentElem = static_cast<Element*> ( aParent );
                const auto& parentAttrs = parentElem->attributes();
                if ( parentAttrs.find ( mAttributeName ) != parentAttrs.end() )
                {
                    mOriginalValue = std::stod ( parentAttrs.at ( mAttributeName ) );
                }
                // Read anchor points for size animations
                if ( parentAttrs.find ( "x" ) != parentAttrs.end() )
                {
                    mAnchorX = std::stod ( parentAttrs.at ( "x" ) );
                }
                else if ( parentAttrs.find ( "cx" ) != parentAttrs.end() )
                {
                    mAnchorX = std::stod ( parentAttrs.at ( "cx" ) );
                }
                if ( parentAttrs.find ( "y" ) != parentAttrs.end() )
                {
                    mAnchorY = std::stod ( parentAttrs.at ( "y" ) );
                }
                else if ( parentAttrs.find ( "cy" ) != parentAttrs.end() )
                {
                    mAnchorY = std::stod ( parentAttrs.at ( "cy" ) );
                }
            }

            // Parse values from "values" attribute or from/to pair
            std::vector<std::string> valueStrings;
            if ( mAttributes.find ( "values" ) != mAttributes.end() )
            {
                valueStrings = SplitValues ( mAttributes.at ( "values" ) );
            }
            else
            {
                if ( mAttributes.find ( "from" ) != mAttributes.end() )
                {
                    valueStrings.push_back ( mAttributes.at ( "from" ) );
                }
                if ( mAttributes.find ( "to" ) != mAttributes.end() )
                {
                    valueStrings.push_back ( mAttributes.at ( "to" ) );
                }
            }

            if ( mIsColorAnimation )
            {
                mColorValues.reserve ( valueStrings.size() );
                for ( const auto& vs : valueStrings )
                {
                    mColorValues.emplace_back ( vs );
                }
            }
            else
            {
                mNumericValues.reserve ( valueStrings.size() );
                for ( const auto& vs : valueStrings )
                {
                    mNumericValues.push_back ( std::stod ( vs ) );
                }
            }
        }

        SVGAnimateElement::~SVGAnimateElement() = default;

        Color SVGAnimateElement::InterpolateColor ( double aProgress ) const
        {
            if ( mColorValues.empty() )
            {
                return Color { 0u };
            }
            if ( mColorValues.size() == 1 )
            {
                return mColorValues[0];
            }
            size_t segments = mColorValues.size() - 1;
            double segProgress = aProgress * segments;
            size_t idx = std::min ( static_cast<size_t> ( segProgress ), segments - 1 );
            double t = segProgress - idx;
            const Color& c0 = mColorValues[idx];
            const Color& c1 = mColorValues[idx + 1];
            return Color
            {
                static_cast<uint8_t> ( c0.a + ( c1.a - c0.a ) * t ),
                static_cast<uint8_t> ( c0.r + ( c1.r - c0.r ) * t ),
                static_cast<uint8_t> ( c0.g + ( c1.g - c0.g ) * t ),
                static_cast<uint8_t> ( c0.b + ( c1.b - c0.b ) * t )
            };
        }

        double SVGAnimateElement::InterpolateNumber ( double aProgress ) const
        {
            if ( mNumericValues.empty() )
            {
                return 0.0;
            }
            if ( mNumericValues.size() == 1 )
            {
                return mNumericValues[0];
            }
            size_t segments = mNumericValues.size() - 1;
            double segProgress = aProgress * segments;
            size_t idx = std::min ( static_cast<size_t> ( segProgress ), segments - 1 );
            double t = segProgress - idx;
            return mNumericValues[idx] + ( mNumericValues[idx + 1] - mNumericValues[idx] ) * t;
        }

        bool SVGAnimateElement::IsGeometryAnimation() const
        {
            return mIsGeometryAnimation;
        }

        bool SVGAnimateElement::IsPathAnimation() const
        {
            return mIsPathAnimation;
        }

        double SVGAnimateElement::GetInterpolatedValue() const
        {
            return InterpolateNumber ( mProgress );
        }

        void SVGAnimateElement::ApplyGeometryToCanvas ( Canvas& aCanvas ) const
        {
            double interpolated = InterpolateNumber ( mProgress );
            if ( mAttributeName == "cx" || mAttributeName == "x" )
            {
                double delta = interpolated - mOriginalValue;
                aCanvas.Transform ( Matrix2x3 { 1.0, 0.0, 0.0, 1.0, delta, 0.0 } );
            }
            else if ( mAttributeName == "cy" || mAttributeName == "y" )
            {
                double delta = interpolated - mOriginalValue;
                aCanvas.Transform ( Matrix2x3 { 1.0, 0.0, 0.0, 1.0, 0.0, delta } );
            }
            else if ( mAttributeName == "width" && mOriginalValue > 0.0 )
            {
                // Anchored scaleX at the element's x position:
                // translate(x,0) * scaleX(s) * translate(-x,0) = matrix(s, 0, 0, 1, x*(1-s), 0)
                double s = interpolated / mOriginalValue;
                aCanvas.Transform ( Matrix2x3 { s, 0.0, 0.0, 1.0, mAnchorX * ( 1.0 - s ), 0.0 } );
            }
            else if ( mAttributeName == "height" && mOriginalValue > 0.0 )
            {
                // Anchored scaleY at the element's y position:
                // translate(0,y) * scaleY(s) * translate(0,-y) = matrix(1, 0, 0, s, 0, y*(1-s))
                double s = interpolated / mOriginalValue;
                aCanvas.Transform ( Matrix2x3 { 1.0, 0.0, 0.0, s, 0.0, mAnchorY * ( 1.0 - s ) } );
            }
            else if ( mAttributeName == "r" && mOriginalValue > 0.0 )
            {
                // Anchored uniform scale at the circle's center (cx, cy):
                // matrix(s, 0, 0, s, cx*(1-s), cy*(1-s))
                double s = interpolated / mOriginalValue;
                aCanvas.Transform ( Matrix2x3 { s, 0.0, 0.0, s, mAnchorX * ( 1.0 - s ), mAnchorY * ( 1.0 - s ) } );
            }
        }

        void SVGAnimateElement::ApplyToCanvas ( Canvas& aCanvas ) const
        {
            if ( !mIsActive )
            {
                return;
            }
            if ( mIsGeometryAnimation )
            {
                ApplyGeometryToCanvas ( aCanvas );
                return;
            }
            if ( mIsColorAnimation )
            {
                Color interpolated = InterpolateColor ( mProgress );
                if ( mAttributeName == "fill" )
                {
                    aCanvas.SetFillColor ( interpolated );
                }
                else if ( mAttributeName == "stroke" )
                {
                    aCanvas.SetStrokeColor ( interpolated );
                }
            }
            else
            {
                double interpolated = InterpolateNumber ( mProgress );
                if ( mAttributeName == "opacity" )
                {
                    aCanvas.SetOpacity ( interpolated );
                }
                else if ( mAttributeName == "fill-opacity" )
                {
                    aCanvas.SetFillOpacity ( interpolated );
                }
                else if ( mAttributeName == "stroke-opacity" )
                {
                    aCanvas.SetStrokeOpacity ( interpolated );
                }
                else if ( mAttributeName == "stroke-width" )
                {
                    aCanvas.SetStrokeWidth ( interpolated );
                }
            }
        }
    }
}
