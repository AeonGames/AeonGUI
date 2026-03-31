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
#ifndef AEONGUI_SVGFILTERELEMENT_H
#define AEONGUI_SVGFILTERELEMENT_H

#include "SVGElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG \<filter\> element (stub — not yet rendered).
         *  @see https://www.w3.org/TR/filter-effects/#InterfaceSVGFilterElement
         */
        class SVGFilterElement : public SVGElement
        {
        public:
            SVGFilterElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGFilterElement() final;
            bool IsDrawEnabled() const final;
        };
    }
}
#endif
