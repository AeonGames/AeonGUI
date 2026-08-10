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
#ifndef AEONGUI_HTMLTEXTAREAELEMENT_H
#define AEONGUI_HTMLTEXTAREAELEMENT_H

#include <cstdint>
#include "HTMLFormControlElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;textarea&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/form-elements.html#htmltextareaelement
         *
         *  A multi-line plain text control.  The element's text content
         *  is the default value; the live value is edited in place and
         *  painted wrapped inside the content box.
         */
        class HTMLTextAreaElement : public HTMLFormControlElement
        {
        public:
            HTMLTextAreaElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLTextAreaElement() final;

            /** @brief Visible height in text lines (`rows`, default 2). */
            AEONGUI_DLL uint32_t rows() const;
            /** @brief Visible width in characters (`cols`, default 20). */
            AEONGUI_DLL uint32_t cols() const;
            /** @brief Placeholder shown while the control is empty. */
            AEONGUI_DLL DOMString placeholder() const;
            /** @brief Whether the value can be edited by the user. */
            AEONGUI_DLL bool readOnly() const;
            /** @brief Value declared in the markup (the text content). */
            AEONGUI_DLL DOMString defaultValue() const;
            /** @brief Caret position as a UTF-8 byte offset into the value. */
            AEONGUI_DLL size_t caretPosition() const;

            AEONGUI_DLL DOMString value() const override;
            AEONGUI_DLL void setValue ( const DOMString& aValue ) override;
            AEONGUI_DLL void Reset() override;
            AEONGUI_DLL bool IsTextEditable() const override;
            AEONGUI_DLL bool HandleKey ( const DOMString& aKey ) override;
            AEONGUI_DLL bool GetIntrinsicContentSize ( float& aWidth, float& aHeight ) const override;

            AEONGUI_DLL void DrawStart ( Canvas& aCanvas ) const override;

        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;

        private:
            /// Copy the markup-declared default into the live value the
            /// first time the control is edited.
            void EnsureValueInitialized();

            DOMString mValue{};
            size_t mCaret{0};
            bool mValueInitialized{false};
        };
    }
}
#endif
