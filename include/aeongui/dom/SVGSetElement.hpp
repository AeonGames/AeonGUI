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
#ifndef AEONGUI_SVGSETELEMENT_H
#define AEONGUI_SVGSETELEMENT_H

#include "SVGAnimationElement.hpp"
#include "aeongui/Color.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SMIL \<set\> element for discrete attribute changes.
         *
         *  Sets an attribute to a fixed value for the duration of the
         *  active interval.  Supports time-based and event-based begin.
         *  @see https://www.w3.org/TR/SVG11/animate.html#SetElement
         */
        class SVGSetElement : public SVGAnimationElement
        {
        public:
            /** @brief Construct an SVGSetElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGSetElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGSetElement() override;
            /** @brief Apply the discrete value change to the canvas.
             *  @param aCanvas The target canvas.
             */
            void ApplyToCanvas ( Canvas& aCanvas ) const override;
        private:
            std::string mToValue;
            bool mIsColorAttribute{false};
            Color mColorValue;
        };
    }
}
#endif
