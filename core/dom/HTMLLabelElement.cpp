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
#include "aeongui/dom/HTMLLabelElement.hpp"
#include "aeongui/dom/HTMLFormControlElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/Canvas.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        HTMLLabelElement::HTMLLabelElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLElement { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLLabelElement::~HTMLLabelElement() = default;

        DOMString HTMLLabelElement::htmlFor() const
        {
            const DOMString* attribute = getAttribute ( "for" );
            return attribute ? *attribute : DOMString{};
        }

        HTMLFormControlElement* HTMLLabelElement::control() const
        {
            const DOMString target = htmlFor();
            if ( !target.empty() )
            {
                if ( const Document * document = ownerDocument() )
                {
                    return dynamic_cast<HTMLFormControlElement*> (
                               document->getElementById ( target ) );
                }
                return nullptr;
            }
            HTMLFormControlElement* first{nullptr};
            const_cast<HTMLLabelElement*> ( this )->TraverseDepthFirstPreOrder (
                [&first] ( Node & aNode )
            {
                if ( first )
                {
                    return;
                }
                first = dynamic_cast<HTMLFormControlElement*> ( &aNode );
            } );
            return first;
        }

        void HTMLLabelElement::DrawStart ( Canvas& aCanvas ) const
        {
            HTMLElement::DrawStart ( aCanvas );
            // Text alone doesn't stamp the pick buffer, so a click
            // anywhere on the label would otherwise miss it.
            PaintHitArea ( aCanvas );
        }
    }
}
