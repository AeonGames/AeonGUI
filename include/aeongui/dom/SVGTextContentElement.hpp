/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGCONTENTELEMENT_H
#define AEONGUI_SVGCONTENTELEMENT_H

#include "SVGGraphicsElement.hpp"
#include "SVGAnimatedLength.hpp"
#include "SVGAnimatedEnumeration.hpp"
#include "DOMPoint.hpp"
#include "DOMRect.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class SVGTextContentElement : public SVGGraphicsElement
        {
        public:
            SVGTextContentElement ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent );
            ~SVGTextContentElement() override;
            const SVGAnimatedLength& textLength() const;
            const SVGAnimatedEnumeration& lengthAdjust() const;

            long getNumberOfChars() const;
            float getComputedTextLength() const;
            float getSubStringLength ( long start, long end ) const;
            DOMPoint getStartPositionOfChar ( long index ) const;
            DOMPoint getEndPositionOfChar ( long index ) const;
            DOMRect getExtentOfChar ( long index ) const;
            float getRotationOfChar ( long index ) const;
            long getCharNumAtPosition ( const DOMPoint& point ) const;
        private:
            /// Helper function to get the text content from all child text nodes
            std::string getTextContent() const;

            SVGAnimatedLength mTextLength;
            SVGAnimatedEnumeration mLengthAdjust;
        };
    }
}
#endif
