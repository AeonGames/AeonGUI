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
#include "aeongui/dom/SVGTextContentElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextContentElement::SVGTextContentElement ( const std::string& aTagName, const AttributeMap& aAttributes, Node* aParent ) : SVGGraphicsElement { aTagName, aAttributes, aParent }
        {
        }
        SVGTextContentElement::~SVGTextContentElement() = default;
        const SVGAnimatedLength& SVGTextContentElement::textLength() const
        {
            return mTextLength;
        }
        long SVGTextContentElement::getNumberOfChars() const
        {
            return 0;
        }
        float SVGTextContentElement::getComputedTextLength() const
        {
            return 0.0f;
        }
        float SVGTextContentElement::getSubStringLength ( long start, long end ) const
        {
            return 0.0f;
        }
        DOMPoint SVGTextContentElement::getStartPositionOfChar ( long index ) const
        {
            return DOMPoint();
        }
        DOMPoint SVGTextContentElement::getEndPositionOfChar ( long index ) const
        {
            return DOMPoint();
        }
        //DOMRect SVGTextContentElement::getExtentOfChar(long index) const
        //{
        //    return DOMRect();
        //}
        float SVGTextContentElement::getRotationOfChar ( long index ) const
        {
            return 0.0f;
        }
        long SVGTextContentElement::getCharNumAtPosition ( const DOMPoint& point ) const
        {
            return 0;
        }

    }
}
