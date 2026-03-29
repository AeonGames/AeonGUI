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
#include <cctype>
#include "aeongui/dom/SVGAnimateMotionElement.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/Matrix2x3.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGAnimateMotionElement::SVGAnimateMotionElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGAnimationElement { aTagName, std::move ( aAttributes ), aParent }
        {
            if ( mAttributes.find ( "path" ) != mAttributes.end() )
            {
                LinearizePath ( mAttributes.at ( "path" ) );
            }
        }

        SVGAnimateMotionElement::~SVGAnimateMotionElement() = default;

        void SVGAnimateMotionElement::LinearizePath ( const std::string& aPathData )
        {
            // Simple path parser supporting M, L, Q, C commands
            double curX = 0.0;
            double curY = 0.0;
            size_t i = 0;
            const size_t len = aPathData.size();

            auto skipWhitespace = [&]()
            {
                while ( i < len && ( aPathData[i] == ' ' || aPathData[i] == ',' || aPathData[i] == '\t' || aPathData[i] == '\n' || aPathData[i] == '\r' ) )
                {
                    ++i;
                }
            };

            auto parseNumber = [&]() -> double
            {
                skipWhitespace();
                size_t start = i;
                if ( i < len && ( aPathData[i] == '-' || aPathData[i] == '+' ) )
                {
                    ++i;
                }
                while ( i < len && ( std::isdigit ( static_cast<unsigned char> ( aPathData[i] ) ) || aPathData[i] == '.' ) )
                {
                    ++i;
                }
                return std::stod ( aPathData.substr ( start, i - start ) );
            };

            auto addPoint = [&] ( double x, double y )
            {
                double dist = 0.0;
                if ( !mPathPoints.empty() )
                {
                    double dx = x - mPathPoints.back().x;
                    double dy = y - mPathPoints.back().y;
                    dist = mPathPoints.back().cumulativeLength + std::sqrt ( dx * dx + dy * dy );
                }
                mPathPoints.push_back ( { x, y, dist } );
            };

            // Sample Bezier curves
            constexpr int SAMPLES = 16;

            while ( i < len )
            {
                skipWhitespace();
                if ( i >= len )
                {
                    break;
                }

                char cmd = aPathData[i];
                if ( std::isalpha ( static_cast<unsigned char> ( cmd ) ) )
                {
                    ++i;
                }
                skipWhitespace();

                switch ( cmd )
                {
                case 'M':
                    curX = parseNumber();
                    curY = parseNumber();
                    addPoint ( curX, curY );
                    break;
                case 'm':
                    curX += parseNumber();
                    curY += parseNumber();
                    addPoint ( curX, curY );
                    break;
                case 'L':
                    curX = parseNumber();
                    curY = parseNumber();
                    addPoint ( curX, curY );
                    break;
                case 'l':
                    curX += parseNumber();
                    curY += parseNumber();
                    addPoint ( curX, curY );
                    break;
                case 'Q':
                {
                    double cpx = parseNumber();
                    double cpy = parseNumber();
                    double endx = parseNumber();
                    double endy = parseNumber();
                    double startx = curX;
                    double starty = curY;
                    for ( int s = 1; s <= SAMPLES; ++s )
                    {
                        double t = static_cast<double> ( s ) / SAMPLES;
                        double u = 1.0 - t;
                        double px = u * u * startx + 2 * u * t * cpx + t * t * endx;
                        double py = u * u * starty + 2 * u * t * cpy + t * t * endy;
                        addPoint ( px, py );
                    }
                    curX = endx;
                    curY = endy;
                    break;
                }
                case 'q':
                {
                    double cpx = curX + parseNumber();
                    double cpy = curY + parseNumber();
                    double endx = curX + parseNumber();
                    double endy = curY + parseNumber();
                    double startx = curX;
                    double starty = curY;
                    for ( int s = 1; s <= SAMPLES; ++s )
                    {
                        double t = static_cast<double> ( s ) / SAMPLES;
                        double u = 1.0 - t;
                        double px = u * u * startx + 2 * u * t * cpx + t * t * endx;
                        double py = u * u * starty + 2 * u * t * cpy + t * t * endy;
                        addPoint ( px, py );
                    }
                    curX = endx;
                    curY = endy;
                    break;
                }
                case 'C':
                {
                    double cp1x = parseNumber();
                    double cp1y = parseNumber();
                    double cp2x = parseNumber();
                    double cp2y = parseNumber();
                    double endx = parseNumber();
                    double endy = parseNumber();
                    double startx = curX;
                    double starty = curY;
                    for ( int s = 1; s <= SAMPLES; ++s )
                    {
                        double t = static_cast<double> ( s ) / SAMPLES;
                        double u = 1.0 - t;
                        double px = u * u * u * startx + 3 * u * u * t * cp1x + 3 * u * t * t * cp2x + t * t * t * endx;
                        double py = u * u * u * starty + 3 * u * u * t * cp1y + 3 * u * t * t * cp2y + t * t * t * endy;
                        addPoint ( px, py );
                    }
                    curX = endx;
                    curY = endy;
                    break;
                }
                case 'c':
                {
                    double cp1x = curX + parseNumber();
                    double cp1y = curY + parseNumber();
                    double cp2x = curX + parseNumber();
                    double cp2y = curY + parseNumber();
                    double endx = curX + parseNumber();
                    double endy = curY + parseNumber();
                    double startx = curX;
                    double starty = curY;
                    for ( int s = 1; s <= SAMPLES; ++s )
                    {
                        double t = static_cast<double> ( s ) / SAMPLES;
                        double u = 1.0 - t;
                        double px = u * u * u * startx + 3 * u * u * t * cp1x + 3 * u * t * t * cp2x + t * t * t * endx;
                        double py = u * u * u * starty + 3 * u * u * t * cp1y + 3 * u * t * t * cp2y + t * t * t * endy;
                        addPoint ( px, py );
                    }
                    curX = endx;
                    curY = endy;
                    break;
                }
                case 'Z':
                case 'z':
                    if ( !mPathPoints.empty() )
                    {
                        addPoint ( mPathPoints[0].x, mPathPoints[0].y );
                        curX = mPathPoints[0].x;
                        curY = mPathPoints[0].y;
                    }
                    break;
                default:
                    // Unknown command, skip
                    break;
                }
            }

            if ( !mPathPoints.empty() )
            {
                mTotalLength = mPathPoints.back().cumulativeLength;
            }
        }

        void SVGAnimateMotionElement::ApplyToCanvas ( Canvas& aCanvas ) const
        {
            if ( !mIsActive || mPathPoints.size() < 2 || mTotalLength <= 0.0 )
            {
                return;
            }

            double targetDist = mProgress * mTotalLength;

            // Find the segment containing the target distance
            size_t idx = 0;
            for ( size_t i = 1; i < mPathPoints.size(); ++i )
            {
                if ( mPathPoints[i].cumulativeLength >= targetDist )
                {
                    idx = i - 1;
                    break;
                }
                idx = i - 1;
            }

            // Interpolate within the segment
            double segStart = mPathPoints[idx].cumulativeLength;
            double segEnd = mPathPoints[idx + 1].cumulativeLength;
            double segLen = segEnd - segStart;
            double t = ( segLen > 0.0 ) ? ( targetDist - segStart ) / segLen : 0.0;

            double x = mPathPoints[idx].x + ( mPathPoints[idx + 1].x - mPathPoints[idx].x ) * t;
            double y = mPathPoints[idx].y + ( mPathPoints[idx + 1].y - mPathPoints[idx].y ) * t;

            // Apply as translation
            aCanvas.Transform ( Matrix2x3
            {
                1.0, 0.0,
                0.0, 1.0,
                x, y
            } );
        }
    }
}
