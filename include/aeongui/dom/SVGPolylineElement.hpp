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
#ifndef AEONGUI_SVGPOLYLINEELEMENT_H
#define AEONGUI_SVGPOLYLINEELEMENT_H

#include "SVGGeometryElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG polyline shape element.
         *  @see https://www.w3.org/TR/SVG2/shapes.html#InterfaceSVGPolylineElement
         */
        class SVGPolylineElement : public SVGGeometryElement
        {
        public:
            /** @brief Construct an SVGPolylineElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGPolylineElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGPolylineElement() final;
        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;
        private:
            void BuildPath();
        };
    }
}
#endif
