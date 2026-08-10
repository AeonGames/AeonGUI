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
#include "aeongui/dom/HTMLFormControlElement.hpp"
#include "aeongui/dom/HTMLFormElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        namespace
        {
            css_computed_style* MutableStyleOf ( const HTMLElement& aElement )
            {
                css_select_results* results = aElement.GetComputedStyles();
                return results ? results->styles[CSS_PSEUDO_ELEMENT_NONE] : nullptr;
            }
        }
        HTMLFormControlElement::HTMLFormControlElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLElement { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLFormControlElement::~HTMLFormControlElement() = default;

        DOMString HTMLFormControlElement::name() const
        {
            const DOMString* attribute = getAttribute ( "name" );
            return attribute ? *attribute : DOMString{};
        }

        HTMLFormElement* HTMLFormControlElement::form() const
        {
            if ( const DOMString * owner = getAttribute ( "form" ) )
            {
                if ( const Document * document = ownerDocument() )
                {
                    return dynamic_cast<HTMLFormElement*> ( document->getElementById ( *owner ) );
                }
                return nullptr;
            }
            for ( Node * node = parentNode(); node != nullptr; node = node->parentNode() )
            {
                if ( auto * candidate = dynamic_cast<HTMLFormElement * > ( node ) )
                {
                    return candidate;
                }
            }
            return nullptr;
        }

        DOMString HTMLFormControlElement::value() const
        {
            const DOMString* attribute = getAttribute ( "value" );
            return attribute ? *attribute : DOMString{};
        }

        void HTMLFormControlElement::setValue ( const DOMString& aValue )
        {
            setAttribute ( "value", aValue );
        }

        void HTMLFormControlElement::Reset() {}

        void HTMLFormControlElement::AppendFormData ( FormData& aFormData ) const
        {
            const DOMString control_name = name();
            if ( control_name.empty() || isDisabled() )
            {
                return;
            }
            aFormData.emplace_back ( control_name, value() );
        }

        void HTMLFormControlElement::Activate() {}

        bool HTMLFormControlElement::IsTextEditable() const
        {
            return false;
        }

        bool HTMLFormControlElement::HandleKey ( const DOMString& )
        {
            return false;
        }

        bool HTMLFormControlElement::HandlePointerDrag ( double, double )
        {
            return false;
        }

        void HTMLFormControlElement::EndPointerDrag() {}

        bool HTMLFormControlElement::GetIntrinsicContentSize ( float&, float& ) const
        {
            return false;
        }

        HTMLFormControlElement::FontSpec HTMLFormControlElement::GetFontSpec() const
        {
            FontSpec spec{};
            if ( css_computed_style * style = MutableStyleOf ( *this ) )
            {
                spec.family = GetCSSFontFamily ( style );
                spec.size   = GetCSSFontSize   ( style );
                spec.weight = GetCSSFontWeight ( style );
                spec.style  = GetCSSFontStyle  ( style );
            }
            return spec;
        }

        void HTMLFormControlElement::PaintCaret ( Canvas& aCanvas, double aX, double aTop, double aHeight ) const
        {
            if ( aHeight <= 0.0 )
            {
                return;
            }
            ColorAttr color{};
            if ( !ResolveTextColor ( color ) )
            {
                return;
            }
            const ColorAttr previous = aCanvas.GetFillColor();
            aCanvas.SetFillColor ( color );
            FillRect ( aCanvas, aX, aTop, aX + 1.0, aTop + aHeight );
            aCanvas.SetFillColor ( previous );
        }

        bool HTMLFormControlElement::canBeDisabled() const
        {
            return true;
        }

        bool HTMLFormControlElement::isDisabled() const
        {
            if ( getAttribute ( "disabled" ) != nullptr )
            {
                return true;
            }
            // A disabled <fieldset> disables every control it contains.
            for ( Node * node = parentNode(); node != nullptr; node = node->parentNode() )
            {
                const auto* element = dynamic_cast<const Element*> ( node );
                if ( element && element->tagName() == "fieldset" &&
                     element->getAttribute ( "disabled" ) != nullptr )
                {
                    return true;
                }
            }
            return false;
        }

        void HTMLFormControlElement::RequestRedraw() const
        {
            if ( Document * document = ownerDocument() )
            {
                document->MarkDirty();
            }
        }
    }
}
