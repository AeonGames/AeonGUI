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
#ifndef AEONGUI_HTMLDIVELEMENT_H
#define AEONGUI_HTMLDIVELEMENT_H

#include "HTMLElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;div&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/grouping-content.html#htmldivelement
         */
        class HTMLDivElement : public HTMLElement
        {
        public:
            HTMLDivElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLDivElement() final;
        };
    }
}
#endif
