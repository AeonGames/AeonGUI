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
#include <iostream>
#include "SVGPathElement.h"
#include "aeongui/Canvas.h"

namespace AeonGUI
{
    namespace Elements
    {
        int ParsePathData ( std::vector<DrawType>& aPath, const char* s );
        SVGPathElement::SVGPathElement ( xmlElementPtr aXmlElementPtr ) : SVGGeometryElement ( aXmlElementPtr )
        {
            if ( HasAttr ( "d" ) )
            {
                std::vector<DrawType> path;
                if ( ParsePathData ( path, GetAttr ( "d" ) ) )
                {
                    if ( HasAttr ( "id" ) )
                    {
                        std::cerr << "Path Id: " << GetAttr ( "id" ) << std::endl;
                    }
                    std::cerr << "Path Data: " << GetAttr ( "d" ) << std::endl;
                }
                mPath.Construct ( path );
            }
        }

        SVGPathElement::~SVGPathElement() = default;
    }
}
