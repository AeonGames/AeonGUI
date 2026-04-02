/*
Copyright (C) 2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGGRADIENTELEMENT_HPP
#define AEONGUI_SVGGRADIENTELEMENT_HPP

#include "SVGElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Base class for SVG gradient elements.
         *  @see https://www.w3.org/TR/SVG2/pservers.html#InterfaceSVGGradientElement
         */
        class SVGGradientElement : public SVGElement
        {
        public:
            /** @brief Construct an SVGGradientElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGGradientElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGGradientElement() override;
        };
    }
}
#endif
