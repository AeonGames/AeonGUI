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
#ifndef AEONGUI_HTMLELEMENT_H
#define AEONGUI_HTMLELEMENT_H

#include "Element.hpp"
#include "aeongui/Color.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Base class for all HTML DOM elements.
         *
         *  Mirrors SVGElement: a thin marker subclass that carries no
         *  HTML-specific state today but anchors the type hierarchy so
         *  layout / rendering passes can dispatch by element family.
         *
         *  @see https://html.spec.whatwg.org/multipage/dom.html#htmlelement
         */
        class HTMLElement : public Element
        {
        public:
            /** @brief XHTML namespace URI. */
            static constexpr const char* kNamespaceURI = "http://www.w3.org/1999/xhtml";

            /** @brief Layout box in CSS pixels, written by HTMLLayoutEngine.
             *
             *  (x, y) is the element's border-box origin in document
             *  coordinates (root at 0,0). (width, height) is the
             *  border-box size.  The content-box fields describe the
             *  inner area after subtracting border + padding on every
             *  side; HTML text and replaced content paint into that
             *  inner box.  NaN-equivalent zero size means "not laid
             *  out yet" or "display: none".
             */
            struct LayoutBox
            {
                float x             {0.0f};
                float y             {0.0f};
                float width         {0.0f};
                float height        {0.0f};
                float contentX      {0.0f};
                float contentY      {0.0f};
                float contentWidth  {0.0f};
                float contentHeight {0.0f};
            };

            /** @brief Construct an HTMLElement.
             *  @param aTagName    Tag name (local name).
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            HTMLElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~HTMLElement() override;

            /** @brief Paint background-color (if any) using the laid-out
             *  border box.  Borders, text, and replaced content come in
             *  later milestones.  Layout must have been computed by
             *  HTMLLayoutEngine first; otherwise the box is 0x0 and
             *  this is a no-op.
             */
            AEONGUI_DLL void DrawStart ( Canvas& aCanvas ) const override;

            /** @brief Get the most recently computed layout box. */
            AEONGUI_DLL const LayoutBox& GetLayoutBox() const noexcept
            {
                return mLayoutBox;
            }

            /** @brief Set the layout box. Called by HTMLLayoutEngine. */
            AEONGUI_DLL void SetLayoutBox ( const LayoutBox& aBox ) noexcept
            {
                mLayoutBox = aBox;
            }

            /// Expose the inherited computed-style accessor publicly so the
            /// HTML layout/render passes can read styles without being a
            /// friend of every Element subclass.
            using Element::GetComputedStyles;

        protected:
            /** @brief Paint background-color and borders over the laid-out
             *  border box.  Split out of DrawStart so form controls can
             *  reuse the CSS box chrome and then paint their own widget
             *  content on top. */
            AEONGUI_DLL void PaintBox ( Canvas& aCanvas ) const;
            /** @brief Lay out and paint this element's inline text content
             *  inside the content box. */
            AEONGUI_DLL void PaintInlineContent ( Canvas& aCanvas ) const;
            /** @brief Fill an axis-aligned rectangle with the canvas' current
             *  fill color.  Degenerate rectangles are ignored. */
            AEONGUI_DLL static void FillRect ( Canvas& aCanvas, double aX0, double aY0,
                                               double aX1, double aY1 );
            /** @brief Fill an axis-aligned ellipse with the canvas' current
             *  fill color, used for radio button chrome. */
            AEONGUI_DLL static void FillEllipse ( Canvas& aCanvas, double aCenterX, double aCenterY,
                                                  double aRadiusX, double aRadiusY );
            /** @brief Fill a closed polygon with the canvas' current fill color.
             *  @param aCanvas     Target canvas.
             *  @param aPoints     Interleaved x/y coordinates.
             *  @param aPointCount Number of (x, y) pairs in @p aPoints. */
            AEONGUI_DLL static void FillPolygon ( Canvas& aCanvas, const double* aPoints,
                                                  size_t aPointCount );
            /** @brief Resolve the computed `color` property to a paint color.
             *  @return false when the color is fully transparent. */
            AEONGUI_DLL bool ResolveTextColor ( ColorAttr& aOut ) const;

        private:
            LayoutBox mLayoutBox{};
        };
    }
}
#endif
