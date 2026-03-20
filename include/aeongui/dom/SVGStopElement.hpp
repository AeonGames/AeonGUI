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
#ifndef AEONGUI_SVGSTOPELEMENT_H
#define AEONGUI_SVGSTOPELEMENT_H

#include "SVGElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {

        /** @brief SVG gradient stop element.
         *  @see https://www.w3.org/TR/SVG2/pservers.html#InterfaceSVGStopElement
         */
        class SVGStopElement : public SVGElement
        {
        public:
            /** @brief Construct an SVGStopElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGStopElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGStopElement() final;
        };
    }
}
#endif
