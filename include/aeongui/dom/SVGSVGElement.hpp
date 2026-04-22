/*
Copyright (C) 2019,2020,2023-2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGSVGELEMENT_H
#define AEONGUI_SVGSVGELEMENT_H

#include "SVGGraphicsElement.hpp"
#include "aeongui/Attribute.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Root SVG container element.
         *  @see https://www.w3.org/TR/SVG2/struct.html#InterfaceSVGSVGElement
         */
        class SVGSVGElement : public SVGGraphicsElement
        {
        public:
            /** @brief Construct an SVGSVGElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGSVGElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGSVGElement() final;
            /** @brief Set up viewport and viewBox, then draw children.
             *  @param aCanvas Target canvas.
             */
            void DrawStart ( Canvas& aCanvas ) const final;
            /** @brief Restore viewport state after children have drawn.
             *  @param aCanvas Target canvas.
             */
            void DrawFinish ( Canvas& aCanvas ) const final;

            /** @brief Box assigned by an HTML layout pass when this
             *  &lt;svg&gt; appears inline inside an HTML document.
             *  Coordinates are in document/canvas pixels.  Width or
             *  height of 0 means "no inline placement" — DrawStart
             *  then falls back to occupying the full canvas as before.
             */
            struct InlineLayoutBox
            {
                double x{}, y{}, width{}, height{};
            };
            AEONGUI_DLL void SetInlineLayoutBox ( const InlineLayoutBox& aBox ) noexcept
            {
                mInlineLayoutBox = aBox;
            }
            AEONGUI_DLL const InlineLayoutBox& GetInlineLayoutBox() const noexcept
            {
                return mInlineLayoutBox;
            }

            /// Intrinsic width/height as authored on the element.
            /// Returns 0 when the corresponding attribute is missing
            /// or expressed as a percentage (which has no intrinsic
            /// meaning outside of a containing block).
            AEONGUI_DLL double GetIntrinsicWidth()  const noexcept
            {
                return ( !mWidthPct  && mWidth  > 0.0 ) ? mWidth  :
                       ( mHasViewBox ? mViewBox.width  : 0.0 );
            }
            AEONGUI_DLL double GetIntrinsicHeight() const noexcept
            {
                return ( !mHeightPct && mHeight > 0.0 ) ? mHeight :
                       ( mHasViewBox ? mViewBox.height : 0.0 );
            }
        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;
        private:
            void ParseAttributes();
            // Attributes
            double mWidth{};
            double mHeight{};
            double mWidthRaw{};
            double mHeightRaw{};
            bool mWidthPct{false};
            bool mHeightPct{false};
            ViewBox mViewBox{};
            PreserveAspectRatio mPreserveAspectRatio{};
            bool mHasViewBox{false};
            InlineLayoutBox mInlineLayoutBox{};
        };
    }
}
#endif
