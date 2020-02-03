/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include "SVGPolygonElement.h"
#include <regex>
#include <iostream>
namespace AeonGUI
{
    namespace Elements
    {
        static const std::regex coord{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]*,?[[:space:]]*(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};
        SVGPolygonElement::SVGPolygonElement ( xmlElementPtr aXmlElementPtr ) : SVGGeometryElement ( aXmlElementPtr )
        {
            std::cout << "Polygon" << std::endl;
            /// https://www.w3.org/TR/SVG/shapes.html#PolygonElement
            if ( HasAttr ( "points" ) )
            {
                std::vector<DrawType> path;
                const char* points = GetAttr ( "points" );
                path.reserve ( ( std::distance ( std::cregex_iterator ( points, points + strlen ( points ) + 1, coord ), std::cregex_iterator() ) * 3 ) + 1 );
                std::cmatch match;
                std::regex_search ( points, match, coord );
                points = match.suffix().first;
                path.emplace_back ( static_cast<uint64_t> ( 'M' ) );
                path.emplace_back ( std::stod ( match[1] ) );
                path.emplace_back ( std::stod ( match[2] ) );
                while ( std::regex_search ( points, match, coord ) )
                {
                    path.emplace_back ( static_cast<uint64_t> ( 'L' ) );
                    path.emplace_back ( std::stod ( match[1] ) );
                    path.emplace_back ( std::stod ( match[2] ) );
                    points = match.suffix().first;
                }
                path.emplace_back ( static_cast<uint64_t> ( 'Z' ) );
                mPath.Construct ( path );
            }
        }
        SVGPolygonElement::~SVGPolygonElement()
        {
        }
    }
}