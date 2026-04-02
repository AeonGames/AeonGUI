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
#ifndef AEONGUI_SVGFEDROPSHADOWELEMENT_H
#define AEONGUI_SVGFEDROPSHADOWELEMENT_H

#include "SVGElement.hpp"
#include "aeongui/Color.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG \<feDropShadow\> filter primitive.
         *  @see https://www.w3.org/TR/filter-effects/#InterfaceSVGFEDropShadowElement
         */
        class SVGFEDropShadowElement : public SVGElement
        {
        public:
            /** @brief Construct an SVGFEDropShadowElement.
             *  @param aTagName    Tag name ("feDropShadow").
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGFEDropShadowElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGFEDropShadowElement() final;
            bool IsDrawEnabled() const final;
            /** @brief Horizontal shadow offset.
             *  @return The dx value.
             */
            double dx() const
            {
                return mDx;
            }
            /** @brief Vertical shadow offset.
             *  @return The dy value.
             */
            double dy() const
            {
                return mDy;
            }
            /** @brief Horizontal Gaussian blur standard deviation.
             *  @return The stdDeviationX value.
             */
            double stdDeviationX() const
            {
                return mStdDeviationX;
            }
            /** @brief Vertical Gaussian blur standard deviation.
             *  @return The stdDeviationY value.
             */
            double stdDeviationY() const
            {
                return mStdDeviationY;
            }
            /** @brief Shadow fill color.
             *  @return The flood color.
             */
            Color floodColor() const
            {
                return mFloodColor;
            }
            /** @brief Shadow opacity [0.0, 1.0].
             *  @return The flood opacity.
             */
            double floodOpacity() const
            {
                return mFloodOpacity;
            }
        private:
            double mDx{2};
            double mDy{2};
            double mStdDeviationX{2};
            double mStdDeviationY{2};
            Color mFloodColor{CSS3Color::black};
            double mFloodOpacity{1.0};
        };
    }
}
#endif
