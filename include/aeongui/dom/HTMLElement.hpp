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
             *  border-box size. NaN means "not laid out yet".
             */
            struct LayoutBox
            {
                float x      {0.0f};
                float y      {0.0f};
                float width  {0.0f};
                float height {0.0f};
            };

            /** @brief Construct an HTMLElement.
             *  @param aTagName    Tag name (local name).
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            HTMLElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~HTMLElement() override;

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

        private:
            LayoutBox mLayoutBox{};
        };
    }
}
#endif
