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
        int ParsePathData ( std::vector<DrawType>& aPath, const char* s, size_t& aEstimate );
        SVGPathElement::SVGPathElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGeometryElement { aTagName, std::move ( aAttributes ), aParent }
        {
            auto it = mAttributes.find ( "d" );
            if ( it != mAttributes.end() )
            {
                size_t estimate = 0;
                if ( ParsePathData ( mPathData, it->second.c_str(), estimate ) )
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
                mPath->Construct ( mPathData, estimate );
                mLastPathD = it->second;
            }
        }

        SVGPathElement::~SVGPathElement() = default;

        void SVGPathElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "d" )
            {
                if ( aValue == mLastPathD )
                {
                    // SMIL/script frequently writes back the same `d`
                    // string every frame; the Flex/Bison parse is
                    // expensive so skip when the source is unchanged.
                    return;
                }
                // Reuse the existing buffer's capacity — ParsePathData
                // appends, so clear() (not resize/reallocate) is what we
                // want.
                mPathData.clear();
                size_t estimate = 0;
                ParsePathData ( mPathData, aValue.c_str(), estimate );
                mPath->Construct ( mPathData, estimate );
                mLastPathD = aValue;
            }
        }
    }
}
