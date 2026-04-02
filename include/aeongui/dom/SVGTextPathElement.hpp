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
#ifndef AEONGUI_SVGTEXTPATHELEMENT_H
#define AEONGUI_SVGTEXTPATHELEMENT_H

#include "SVGTextContentElement.hpp"
#include "SVGAnimatedString.hpp"
#include "SVGAnimatedLength.hpp"
#include "SVGAnimatedEnumeration.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG textPath element for rendering text along a path.
         *  @see https://www.w3.org/TR/SVG2/text.html#InterfaceSVGTextPathElement
         */
        class SVGTextPathElement : public SVGTextContentElement
        {
        public:
            /** @brief Construct an SVG textPath element.
             *  @param aTagName Element tag name.
             *  @param aAttributes Initial attribute map.
             *  @param aParent Parent node.
             */
            SVGTextPathElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGTextPathElement() override;
            void DrawStart ( Canvas& aCanvas ) const override;

            /** @brief Get the path reference attribute.
             *  @return Animated href attribute.
             */
            const SVGAnimatedString& href() const;
            /** @brief Get the start offset along the referenced path.
             *  @return Animated startOffset attribute.
             */
            const SVGAnimatedLength& startOffset() const;
            /** @brief Get text path placement method.
             *  @return Animated method attribute.
             */
            const SVGAnimatedEnumeration& method() const;
            /** @brief Get text path spacing mode.
             *  @return Animated spacing attribute.
             */
            const SVGAnimatedEnumeration& spacing() const;

        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;

        private:
            void ParseAttributes();
            SVGAnimatedString mHref;
            SVGAnimatedLength mStartOffset;
            SVGAnimatedEnumeration mMethod;
            SVGAnimatedEnumeration mSpacing;
            bool mSideRight{false};  ///< true when side="right" (reverse path).
            int mTextAnchor{0};      ///< 0=start, 1=middle, 2=end.
        };
    }
}
#endif
