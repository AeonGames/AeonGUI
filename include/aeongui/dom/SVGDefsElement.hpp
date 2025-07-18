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
#ifndef AEONGUI_SVGDEFSELEMENT_H
#define AEONGUI_SVGDEFSELEMENT_H

#include "SVGGraphicsElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class SVGDefsElement : public SVGGraphicsElement
        {
        public:
            SVGDefsElement ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent );
            ~SVGDefsElement() final;
            bool IsDrawEnabled() const final;
        };
    }
}
#endif
