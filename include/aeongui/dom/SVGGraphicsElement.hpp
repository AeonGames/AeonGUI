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
#ifndef AEONGUI_SVGGRAPHICSELEMENT_H
#define AEONGUI_SVGGRAPHICSELEMENT_H

#include "SVGElement.hpp"
#include "aeongui/Matrix2x3.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Base class for SVG elements that can be rendered with transformations.
         *  @see https://www.w3.org/TR/SVG2/types.html#InterfaceSVGGraphicsElement
         */
        class SVGGraphicsElement : public SVGElement
        {
        public:
            /** @brief Construct an SVGGraphicsElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGGraphicsElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGGraphicsElement() override;
            /** @brief Apply the element's transform and begin drawing.
             *  If a filter attribute is present, begins offscreen capture.
             *  @param aCanvas Target canvas.
             */
            void DrawStart ( Canvas& aCanvas ) const override;
            /** @brief Finish drawing and apply any filter effects.
             *  @param aCanvas Target canvas.
             */
            void DrawFinish ( Canvas& aCanvas ) const override;
        private:
            Matrix2x3 mTransform{};
            mutable Element* mFilterElement{nullptr};
        };
    }
}
#endif
