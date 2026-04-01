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
#include "aeongui/dom/SVGFEDropShadowElement.hpp"
#include <cstdlib>
#include <cstring>

namespace AeonGUI
{
    namespace DOM
    {
        SVGFEDropShadowElement::SVGFEDropShadowElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGElement ( aTagName, std::move ( aAttributes ), aParent )
        {
            auto parseDouble = [this] ( const char* name, double fallback ) -> double
            {
                const DOMString* val = getAttribute ( name );
                if ( val && !val->empty() )
                {
                    char* end{};
                    double v = strtod ( val->c_str(), &end );
                    if ( end != val->c_str() )
                    {
                        return v;
                    }
                }
                return fallback;
            };
            mDx = parseDouble ( "dx", 2 );
            mDy = parseDouble ( "dy", 2 );
            // stdDeviation can be "x" or "x y"
            const DOMString* stdDev = getAttribute ( "stdDeviation" );
            if ( stdDev && !stdDev->empty() )
            {
                char* end{};
                double sx = strtod ( stdDev->c_str(), &end );
                if ( end != stdDev->c_str() )
                {
                    mStdDeviationX = sx;
                    mStdDeviationY = sx;
                    // Check for second value
                    while ( *end == ' ' || *end == ',' )
                    {
                        ++end;
                    }
                    if ( *end != '\0' )
                    {
                        char* end2{};
                        double sy = strtod ( end, &end2 );
                        if ( end2 != end )
                        {
                            mStdDeviationY = sy;
                        }
                    }
                }
            }
            mFloodOpacity = parseDouble ( "flood-opacity", 1.0 );
            const DOMString* colorStr = getAttribute ( "flood-color" );
            if ( colorStr && !colorStr->empty() )
            {
                uint32_t colorVal{};
                if ( Color::IsColor ( *colorStr, &colorVal ) )
                {
                    mFloodColor = Color{colorVal};
                }
            }
        }
        SVGFEDropShadowElement::~SVGFEDropShadowElement() = default;
        bool SVGFEDropShadowElement::IsDrawEnabled() const
        {
            return false;
        }
    }
}
