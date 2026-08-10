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
#ifndef AEONGUI_HTMLBUTTONELEMENT_H
#define AEONGUI_HTMLBUTTONELEMENT_H

#include "HTMLFormControlElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;button&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/form-elements.html#htmlbuttonelement
         *
         *  Unlike the button-flavoured &lt;input&gt; types the label comes
         *  from the element's children, so layout and paint fall through
         *  to the regular HTMLElement text machinery; only the activation
         *  behavior is implemented here.
         */
        class HTMLButtonElement : public HTMLFormControlElement
        {
        public:
            /** @brief Normalized `type` attribute value. */
            enum class Type
            {
                Submit, ///< type="submit", also the default.
                Reset,  ///< type="reset".
                Button  ///< type="button", no default behavior.
            };

            HTMLButtonElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLButtonElement() final;

            /** @brief Normalized button type. */
            AEONGUI_DLL Type type() const;
            AEONGUI_DLL void Activate() override;
            AEONGUI_DLL bool HandleKey ( const DOMString& aKey ) override;
            AEONGUI_DLL void AppendFormData ( FormData& aFormData ) const override;

        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;

        private:
            Type mType{Type::Submit};
        };
    }
}
#endif
