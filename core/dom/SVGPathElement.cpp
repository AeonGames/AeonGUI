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
#include "aeongui/dom/SVGPathElement.hpp"
#include "aeongui/Canvas.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        int ParsePathData ( std::vector<DrawType>& aPath, const char* s );
        SVGPathElement::SVGPathElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, std::move ( aAttributes ), aParent }
        {
            if ( mAttributes.find ( "d" ) != mAttributes.end() )
            {
                std::vector<DrawType> path;
                if ( ParsePathData ( path, mAttributes.at ( "d" ).c_str() ) )
                {
#if 0
                    auto id = GetAttribute ( "id" );
                    if ( std::holds_alternative<std::string> ( id ) )
                    {
                        std::cerr << "Path Id: " << std::get<std::string> ( id ) << std::endl;
                    }
                    std::cerr << "Path Data: " << std::get<std::string> ( d ) << std::endl;
#endif
                }
                mPath.Construct ( path );
            }
        }

        SVGPathElement::~SVGPathElement() = default;

        void SVGPathElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "d" )
            {
                std::vector<DrawType> path;
                ParsePathData ( path, aValue.c_str() );
                mPath.Construct ( path );
            }
        }
    }
}
