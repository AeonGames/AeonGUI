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
#ifndef AEONGUI_HTMLFIELDSETELEMENT_H
#define AEONGUI_HTMLFIELDSETELEMENT_H

#include "HTMLElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;fieldset&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/form-elements.html#htmlfieldsetelement
         *
         *  A `disabled` fieldset disables every control it contains; that
         *  propagation lives in HTMLFormControlElement::isDisabled().
         */
        class HTMLFieldSetElement : public HTMLElement
        {
        public:
            HTMLFieldSetElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLFieldSetElement() final;

            AEONGUI_DLL bool canBeDisabled() const override;
            AEONGUI_DLL bool isDisabled() const override;
        };

        /** @brief HTML &lt;legend&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/form-elements.html#htmllegendelement
         */
        class HTMLLegendElement : public HTMLElement
        {
        public:
            HTMLLegendElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLLegendElement() final;
        };
    }
}
#endif
