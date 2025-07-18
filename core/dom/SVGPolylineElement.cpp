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
#include "aeongui/dom/SVGPolylineElement.hpp"
#include <regex>
#include <iostream>

namespace AeonGUI
{
    namespace DOM
    {
        static const std::regex coord{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]*,?[[:space:]]*(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};
        SVGPolylineElement::SVGPolylineElement ( const std::string& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, aAttributes, aParent }
        {
            std::cout << "Polyline" << std::endl;
            /// https://www.w3.org/TR/SVG/shapes.html#PolylineElement
            if ( aAttributes.find ( "points" ) != aAttributes.end() )
            {
                std::vector<DrawType> path;
                const std::string& points = aAttributes.at ( "points" );
                auto it = std::sregex_iterator ( points.begin(), points.end(), coord );
                path.reserve ( ( std::distance ( it, std::sregex_iterator() ) * 3 ) );
                std::smatch match = *it;
                path.emplace_back ( static_cast<uint64_t> ( 'M' ) );
                path.emplace_back ( std::stod ( match[1] ) );
                path.emplace_back ( std::stod ( match[2] ) );
                for ( std::sregex_iterator i = ++it; i != std::sregex_iterator(); ++i )
                {
                    match = *i;
                    path.emplace_back ( static_cast<uint64_t> ( 'L' ) );
                    path.emplace_back ( std::stod ( match[1] ) );
                    path.emplace_back ( std::stod ( match[2] ) );
                }
                mPath.Construct ( path );
            }
        }
        SVGPolylineElement::~SVGPolylineElement() = default;
    }
}
