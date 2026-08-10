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
#ifndef AEONGUI_HTMLLABELELEMENT_H
#define AEONGUI_HTMLLABELELEMENT_H

#include "HTMLElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class HTMLFormControlElement;

        /** @brief HTML &lt;label&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/forms.html#htmllabelelement
         */
        class HTMLLabelElement : public HTMLElement
        {
        public:
            HTMLLabelElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLLabelElement() final;

            /** @brief Value of the `for` attribute (empty when absent). */
            AEONGUI_DLL DOMString htmlFor() const;
            /** @brief The labeled control: the `for` target if present,
             *  otherwise the first form control descendant. */
            AEONGUI_DLL HTMLFormControlElement* control() const;

            AEONGUI_DLL void DrawStart ( Canvas& aCanvas ) const override;
        };
    }
}
#endif
