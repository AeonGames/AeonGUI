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
#include "aeongui/dom/HTMLButtonElement.hpp"
#include "aeongui/dom/HTMLFormElement.hpp"
#include "AsciiCase.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        namespace
        {
            HTMLButtonElement::Type ParseButtonType ( const DOMString& aValue )
            {
                if ( EqualsIgnoreCase ( aValue, "reset" ) )
                {
                    return HTMLButtonElement::Type::Reset;
                }
                else if ( EqualsIgnoreCase ( aValue, "button" ) )
                {
                    return HTMLButtonElement::Type::Button;
                }
                // Missing or invalid values default to submit.
                return HTMLButtonElement::Type::Submit;
            }
        }

        HTMLButtonElement::HTMLButtonElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLFormControlElement { aTagName, std::move ( aAttributes ), aParent }
        {
            const DOMString* type_attribute = getAttribute ( "type" );
            mType = type_attribute ? ParseButtonType ( *type_attribute ) : Type::Submit;
        }

        HTMLButtonElement::~HTMLButtonElement() = default;

        HTMLButtonElement::Type HTMLButtonElement::type() const
        {
            return mType;
        }

        void HTMLButtonElement::Activate()
        {
            if ( isDisabled() )
            {
                return;
            }
            HTMLFormElement* owner = form();
            if ( !owner )
            {
                return;
            }
            switch ( mType )
            {
            case Type::Submit:
                owner->submit();
                break;
            case Type::Reset:
                owner->reset();
                break;
            case Type::Button:
                break;
            }
        }

        bool HTMLButtonElement::HandleKey ( const DOMString& aKey )
        {
            if ( aKey != " " && aKey != "Enter" )
            {
                return false;
            }
            Activate();
            return true;
        }

        void HTMLButtonElement::AppendFormData ( FormData& ) const
        {
            // A button is only a successful control when it is the
            // submitter, which the DOM-only submit path does not model.
        }

        void HTMLButtonElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            HTMLFormControlElement::onAttributeChanged ( aName, aValue );
            if ( aName == "type" )
            {
                mType = ParseButtonType ( aValue );
            }
            else if ( aName == "disabled" )
            {
                ReselectCSS();
            }
        }
    }
}
