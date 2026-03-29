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
#include <cmath>
#include <sstream>
#include <algorithm>
#include "aeongui/dom/SVGAnimateTransformElement.hpp"
#include "aeongui/Canvas.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGAnimateTransformElement::SVGAnimateTransformElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGAnimationElement { aTagName, std::move ( aAttributes ), aParent }
        {
            // Parse type
            if ( mAttributes.find ( "type" ) != mAttributes.end() )
            {
                const auto& type = mAttributes.at ( "type" );
                if ( type == "translate" )
                {
                    mTransformType = TransformType::Translate;
                }
                else if ( type == "scale" )
                {
                    mTransformType = TransformType::Scale;
                }
                else if ( type == "rotate" )
                {
                    mTransformType = TransformType::Rotate;
                }
                else if ( type == "skewX" )
                {
                    mTransformType = TransformType::SkewX;
                }
                else if ( type == "skewY" )
                {
                    mTransformType = TransformType::SkewY;
                }
            }

            // Parse additive
            if ( mAttributes.find ( "additive" ) != mAttributes.end() )
            {
                mAdditive = ( mAttributes.at ( "additive" ) == "sum" );
            }

            // Parse keyframes from values or from/to
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

            mKeyframes.reserve ( valueStrings.size() );
            for ( const auto& vs : valueStrings )
            {
                mKeyframes.push_back ( ParseNumbers ( vs ) );
            }
        }

        SVGAnimateTransformElement::~SVGAnimateTransformElement() = default;

        std::vector<double> SVGAnimateTransformElement::ParseNumbers ( const std::string& aStr )
        {
            std::vector<double> result;
            std::istringstream stream ( aStr );
            double val{};
            while ( stream >> val )
            {
                result.push_back ( val );
                // Skip commas
                if ( stream.peek() == ',' )
                {
                    stream.ignore();
                }
            }
            return result;
        }

        Matrix2x3 SVGAnimateTransformElement::ComputeTransform ( double aProgress ) const
        {
            if ( mKeyframes.empty() )
            {
                return Matrix2x3{};
            }

            // Interpolate between keyframes
            std::vector<double> interpolated;
            if ( mKeyframes.size() == 1 )
            {
                interpolated = mKeyframes[0];
            }
            else
            {
                size_t segments = mKeyframes.size() - 1;
                double segProgress = aProgress * segments;
                size_t idx = std::min ( static_cast<size_t> ( segProgress ), segments - 1 );
                double t = segProgress - idx;

                const auto& k0 = mKeyframes[idx];
                const auto& k1 = mKeyframes[idx + 1];
                size_t count = std::max ( k0.size(), k1.size() );
                interpolated.resize ( count, 0.0 );
                for ( size_t i = 0; i < count; ++i )
                {
                    double v0 = ( i < k0.size() ) ? k0[i] : 0.0;
                    double v1 = ( i < k1.size() ) ? k1[i] : 0.0;
                    interpolated[i] = v0 + ( v1 - v0 ) * t;
                }
            }

            // Build transform matrix based on type
            constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;

            switch ( mTransformType )
            {
            case TransformType::Rotate:
            {
                double angle = interpolated.size() > 0 ? interpolated[0] * DEG_TO_RAD : 0.0;
                double cx = interpolated.size() > 1 ? interpolated[1] : 0.0;
                double cy = interpolated.size() > 2 ? interpolated[2] : 0.0;
                double cosA = std::cos ( angle );
                double sinA = std::sin ( angle );
                // Matrix2x3(xx=a, yx=b, xy=c, yy=d, x0=e, y0=f)
                // CSS rotation: a=cos, b=sin, c=-sin, d=cos
                // With center: translate(cx,cy) * rotate(angle) * translate(-cx,-cy)
                return Matrix2x3
                {
                    cosA, sinA,
                    -sinA, cosA,
                    cx - cosA * cx + sinA * cy,
                    cy - sinA * cx - cosA * cy
                };
            }
            case TransformType::Scale:
            {
                double sx = interpolated.size() > 0 ? interpolated[0] : 1.0;
                double sy = interpolated.size() > 1 ? interpolated[1] : sx;
                return Matrix2x3
                {
                    sx, 0.0,
                    0.0, sy,
                    0.0, 0.0
                };
            }
            case TransformType::Translate:
            {
                double tx = interpolated.size() > 0 ? interpolated[0] : 0.0;
                double ty = interpolated.size() > 1 ? interpolated[1] : 0.0;
                return Matrix2x3
                {
                    1.0, 0.0,
                    0.0, 1.0,
                    tx, ty
                };
            }
            case TransformType::SkewX:
            {
                // CSS skewX: matrix(1, 0, tan(a), 1, 0, 0)
                double angle = interpolated.size() > 0 ? interpolated[0] * DEG_TO_RAD : 0.0;
                return Matrix2x3
                {
                    1.0, 0.0,
                    std::tan ( angle ), 1.0,
                    0.0, 0.0
                };
            }
            case TransformType::SkewY:
            {
                // CSS skewY: matrix(1, tan(a), 0, 1, 0, 0)
                double angle = interpolated.size() > 0 ? interpolated[0] * DEG_TO_RAD : 0.0;
                return Matrix2x3
                {
                    1.0, std::tan ( angle ),
                    0.0, 1.0,
                    0.0, 0.0
                };
            }
            }
            return Matrix2x3{};
        }

        void SVGAnimateTransformElement::ApplyToCanvas ( Canvas& aCanvas ) const
        {
            if ( !mIsActive )
            {
                return;
            }
            aCanvas.Transform ( ComputeTransform ( mProgress ) );
        }
    }
}
