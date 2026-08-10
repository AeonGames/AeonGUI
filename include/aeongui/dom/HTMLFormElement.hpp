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
#ifndef AEONGUI_HTMLFORMELEMENT_H
#define AEONGUI_HTMLFORMELEMENT_H

#include <vector>
#include "HTMLElement.hpp"
#include "HTMLFormControlElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;form&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/forms.html#htmlformelement
         *
         *  AeonGUI has no networking stack, so submission stops at the
         *  DOM: submit() fires a cancelable `submit` event carrying the
         *  entry list built from the form's controls, and the host
         *  application decides what to do with it.  reset() likewise
         *  fires a cancelable `reset` event before restoring defaults.
         */
        class HTMLFormElement : public HTMLElement
        {
        public:
            HTMLFormElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLFormElement() final;

            /** @brief Value of the `action` attribute (empty when absent). */
            AEONGUI_DLL DOMString action() const;
            /** @brief Value of the `method` attribute, lowercased,
             *  defaulting to "get". */
            AEONGUI_DLL DOMString method() const;
            /** @brief Value of the `name` attribute (empty when absent). */
            AEONGUI_DLL DOMString name() const;

            /** @brief All form controls owned by this form, in tree order. */
            AEONGUI_DLL std::vector<HTMLFormControlElement*> elements() const;
            /** @brief Build the submission entry list from the owned controls. */
            AEONGUI_DLL HTMLFormControlElement::FormData GetFormData() const;

            /** @brief Fire a cancelable `submit` event on the form.
             *  @return true when the event went unprevented. */
            AEONGUI_DLL bool submit();
            /** @brief Fire a cancelable `reset` event and, when it goes
             *  unprevented, restore every owned control to its default. */
            AEONGUI_DLL void reset();
        };
    }
}
#endif
