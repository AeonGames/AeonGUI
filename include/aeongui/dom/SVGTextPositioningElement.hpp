/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGTEXTPOSITIONINGELEMENT_H
#define AEONGUI_SVGTEXTPOSITIONINGELEMENT_H

#include "SVGTextContentElement.hpp"
#include "SVGAnimatedLengthList.hpp"
#include "SVGAnimatedNumberList.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG element with positioning attributes (x, y, dx, dy, rotate).
         *  @see https://www.w3.org/TR/SVG2/text.html#InterfaceSVGTextPositioningElement
         */
        class SVGTextPositioningElement : public SVGTextContentElement
        {
        public:
            /** @brief Construct an SVGTextPositioningElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGTextPositioningElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGTextPositioningElement() override;

            /** @name SVGTextPositioningElement interface
             *  @{ */
            /** @brief Get the animated x position list.
             *  @return The animated x length list. */
            const SVGAnimatedLengthList& x() const;
            /** @brief Get the animated y position list.
             *  @return The animated y length list. */
            const SVGAnimatedLengthList& y() const;
            /** @brief Get the animated dx offset list.
             *  @return The animated dx length list. */
            const SVGAnimatedLengthList& dx() const;
            /** @brief Get the animated dy offset list.
             *  @return The animated dy length list. */
            const SVGAnimatedLengthList& dy() const;
            /** @brief Get the animated rotation list.
             *  @return The animated rotation number list. */
            const SVGAnimatedNumberList& rotate() const;
            /** @} */

        private:
            /// Helper methods for parsing attribute values
            void parsePositioningAttributes ( const AttributeMap& aAttributes );
            void parseLengthList ( const DOMString& value, SVGLengthList& lengthList );
            void parseNumberList ( const DOMString& value, SVGNumberList& numberList );

            SVGAnimatedLengthList mX;
            SVGAnimatedLengthList mY;
            SVGAnimatedLengthList mDx;
            SVGAnimatedLengthList mDy;
            SVGAnimatedNumberList mRotate;
        };
    }
}
#endif
