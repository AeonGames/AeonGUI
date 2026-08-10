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
#include <algorithm>
#include <cctype>
#include "aeongui/dom/HTMLFormElement.hpp"
#include "aeongui/dom/Event.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        HTMLFormElement::HTMLFormElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLElement { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLFormElement::~HTMLFormElement() = default;

        DOMString HTMLFormElement::action() const
        {
            const DOMString* attribute = getAttribute ( "action" );
            return attribute ? *attribute : DOMString{};
        }

        DOMString HTMLFormElement::method() const
        {
            const DOMString* attribute = getAttribute ( "method" );
            if ( !attribute )
            {
                return DOMString{"get"};
            }
            DOMString lowered{*attribute};
            std::transform ( lowered.begin(), lowered.end(), lowered.begin(),
                             [] ( unsigned char c )
            {
                return static_cast<char> ( std::tolower ( c ) );
            } );
            return lowered;
        }

        DOMString HTMLFormElement::name() const
        {
            const DOMString* attribute = getAttribute ( "name" );
            return attribute ? *attribute : DOMString{};
        }

        std::vector<HTMLFormControlElement*> HTMLFormElement::elements() const
        {
            std::vector<HTMLFormControlElement*> controls;
            const_cast<HTMLFormElement*> ( this )->TraverseDepthFirstPreOrder (
                [this, &controls] ( Node & aNode )
            {
                auto* control = dynamic_cast<HTMLFormControlElement*> ( &aNode );
                if ( control && control->form() == this )
                {
                    controls.push_back ( control );
                }
            } );
            return controls;
        }

        HTMLFormControlElement::FormData HTMLFormElement::GetFormData() const
        {
            HTMLFormControlElement::FormData data;
            for ( const HTMLFormControlElement * control : elements() )
            {
                control->AppendFormData ( data );
            }
            return data;
        }

        bool HTMLFormElement::submit()
        {
            Event submitEvent ( "submit", EventInit{true, true, false} );
            return dispatchEvent ( submitEvent );
        }

        void HTMLFormElement::reset()
        {
            Event resetEvent ( "reset", EventInit{true, true, false} );
            if ( !dispatchEvent ( resetEvent ) )
            {
                return;
            }
            for ( HTMLFormControlElement * control : elements() )
            {
                control->Reset();
            }
        }
    }
}
