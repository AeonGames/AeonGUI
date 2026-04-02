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
#ifndef AEONGUI_SVGCONTENTELEMENT_H
#define AEONGUI_SVGCONTENTELEMENT_H

#include "SVGGraphicsElement.hpp"
#include "SVGAnimatedLength.hpp"
#include "SVGAnimatedEnumeration.hpp"
#include "DOMPoint.hpp"
#include "DOMRect.hpp"
#ifdef AEONGUI_USE_SKIA
#include "aeongui/SkiaTextLayout.hpp"
#else
#include "aeongui/PangoTextLayout.hpp"
#endif
#include <memory>

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Base class for SVG elements that render text.
         *  @see https://www.w3.org/TR/SVG2/text.html#InterfaceSVGTextContentElement
         */
        class SVGTextContentElement : public SVGGraphicsElement
        {
        public:
            /** @brief Construct an SVGTextContentElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGTextContentElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGTextContentElement() override;
            /** @brief Get the animated text length.
             *  @return The animated text length. */
            const SVGAnimatedLength& textLength() const;
            /** @brief Get the animated length adjust enumeration.
             *  @return The animated length adjust value. */
            const SVGAnimatedEnumeration& lengthAdjust() const;

            /** @brief Get the total number of characters.
             *  @return The character count. */
            long getNumberOfChars() const;
            /** @brief Get the computed total advance width of the text.
             *  @return The text advance width. */
            float getComputedTextLength() const;
            /** @brief Get the advance width of a substring.
             *  @param start Start character index.
             *  @param end   End character index.
             *  @return Advance width of the substring.
             */
            float getSubStringLength ( long start, long end ) const;
            /** @brief Get the start position of a character.
             *  @param index Character index.
             *  @return Start position as a DOMPoint.
             */
            DOMPoint getStartPositionOfChar ( long index ) const;
            /** @brief Get the end position of a character.
             *  @param index Character index.
             *  @return End position as a DOMPoint.
             */
            DOMPoint getEndPositionOfChar ( long index ) const;
            /** @brief Get the bounding box of a character.
             *  @param index Character index.
             *  @return Bounding rectangle.
             */
            DOMRect getExtentOfChar ( long index ) const;
            /** @brief Get the rotation of a character.
             *  @param index Character index.
             *  @return Rotation angle in degrees.
             */
            float getRotationOfChar ( long index ) const;
            /** @brief Get the character index at a point.
             *  @param point Position to query.
             *  @return Character index, or -1 if none.
             */
            long getCharNumAtPosition ( const DOMPoint& point ) const;
        protected:
            /// Access the internal text layout for subclass use.
            /// @return Reference to the TextLayout.
#ifdef AEONGUI_USE_SKIA
            SkiaTextLayout& GetTextLayout() const;
#else
            PangoTextLayout& GetTextLayout() const;
#endif
            /** @brief Get concatenated text content from child text nodes.
             *  @return UTF-8 text collected from this node subtree.
             */
            std::string getTextContent() const;
        private:
            /// Ensure the text layout is up to date with current text and font.
            void syncTextLayout() const;

            SVGAnimatedLength mTextLength;
            SVGAnimatedEnumeration mLengthAdjust;
#ifdef AEONGUI_USE_SKIA
            mutable SkiaTextLayout mTextLayout;
#else
            mutable PangoTextLayout mTextLayout;
#endif
        };
    }
}
#endif
