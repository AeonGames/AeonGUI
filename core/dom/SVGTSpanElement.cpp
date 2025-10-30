/*
Copyright (C) 2020,2024,2025 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGTSpanElement.hpp"
#include "aeongui/dom/Text.hpp"
#include <iostream>
namespace AeonGUI
{
    namespace DOM
    {
        SVGTSpanElement::SVGTSpanElement ( const std::string& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGTextPositioningElement { aTagName, aAttributes, aParent }
        {
        }
        SVGTSpanElement::~SVGTSpanElement() = default;

        void SVGTSpanElement::OnLoad()
        {
            std::cout << "Loading SVGTSpanElement" << std::endl;
            for ( auto& child : this->childNodes() )
            {
                if ( child->nodeType() == Node::TEXT_NODE )
                {
                    const Text* textNode = static_cast<const Text*> ( child );
                    std::cout << "Text content: " << textNode->wholeText() << std::endl;
                }
            }
        }

        void SVGTSpanElement::OnUnload()
        {
            std::cout << "Unloading SVGTSpanElement" << std::endl;
        }
    }
}
