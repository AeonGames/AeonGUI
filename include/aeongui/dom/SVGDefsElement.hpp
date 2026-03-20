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
#ifndef AEONGUI_SVGDEFSELEMENT_H
#define AEONGUI_SVGDEFSELEMENT_H

#include "SVGGraphicsElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Container for referenced SVG elements (not rendered directly).
         *  @see https://www.w3.org/TR/SVG2/struct.html#InterfaceSVGDefsElement
         */
        class SVGDefsElement : public SVGGraphicsElement
        {
        public:
            /** @brief Construct an SVGDefsElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGDefsElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGDefsElement() final;
            /** @brief Always returns false; defs children are not drawn directly. */
            bool IsDrawEnabled() const final;
        };
    }
}
#endif
