/*
Copyright (C) 2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGGraphicsElement.hpp"
#include "aeongui/dom/SVGFilterElement.hpp"
#include "aeongui/dom/SVGFEDropShadowElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/Canvas.hpp"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        SVGGraphicsElement::SVGGraphicsElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGElement { aTagName, std::move ( aAttributes ), aParent }
        {
            css_select_results* results{ GetComputedStyles() };
            css_matrix transform{};
            if ( ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] ) &&
                 ( css_computed_transform ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &transform ) != CSS_TRANSFORM_NONE ) )
            {
                mTransform =
                {
                    FIXTOFLT ( transform.m[0] ), FIXTOFLT ( transform.m[1] ),
                    FIXTOFLT ( transform.m[2] ), FIXTOFLT ( transform.m[3] ),
                    FIXTOFLT ( transform.m[4] ), FIXTOFLT ( transform.m[5] )
                };
            }
        }
        SVGGraphicsElement::~SVGGraphicsElement() = default;

        void SVGGraphicsElement::DrawStart ( Canvas& aCanvas ) const
        {
            aCanvas.Transform ( mTransform );
            ApplyChildTransformAnimations ( aCanvas );

            // Check for filter="url(#id)" attribute — skip during hit testing
            mHasFilter = false;
            if ( !aCanvas.IsHitTesting() )
            {
                const DOMString* filterAttr = getAttribute ( "filter" );
                if ( filterAttr && !filterAttr->empty() )
                {
                    if ( filterAttr->compare ( 0, 5, "url(#" ) == 0 && filterAttr->back() == ')' )
                    {
                        std::string filterId = filterAttr->substr ( 5, filterAttr->size() - 6 );
                        Document* doc = ownerDocument();
                        if ( doc )
                        {
                            Element* filterElem = doc->getElementById ( filterId );
                            if ( filterElem && filterElem->tagName() == "filter" )
                            {
                                mHasFilter = true;
                                aCanvas.PushGroup();
                            }
                        }
                    }
                }
            }
        }

        void SVGGraphicsElement::DrawFinish ( Canvas& aCanvas ) const
        {
            if ( !mHasFilter )
            {
                return;
            }

            const DOMString* filterAttr = getAttribute ( "filter" );
            if ( !filterAttr || filterAttr->empty() )
            {
                aCanvas.PopGroup();
                return;
            }

            // Re-resolve the filter element to access its primitives
            if ( filterAttr->compare ( 0, 5, "url(#" ) != 0 || filterAttr->back() != ')' )
            {
                aCanvas.PopGroup();
                return;
            }
            std::string filterId = filterAttr->substr ( 5, filterAttr->size() - 6 );
            Document* doc = ownerDocument();
            Element* filterElem = doc ? doc->getElementById ( filterId ) : nullptr;
            if ( !filterElem || filterElem->tagName() != "filter" )
            {
                aCanvas.PopGroup();
                return;
            }

            // Iterate child filter primitives and apply them
            bool applied = false;
            for ( const auto& child : filterElem->childNodes() )
            {
                if ( child->nodeType() != Node::ELEMENT_NODE )
                {
                    continue;
                }
                Element* primitive = static_cast<Element*> ( child.get() );
                if ( primitive->tagName() == "feDropShadow" )
                {
                    const SVGFEDropShadowElement* dropShadow =
                        static_cast<const SVGFEDropShadowElement*> ( primitive );
                    aCanvas.ApplyDropShadow (
                        dropShadow->dx(), dropShadow->dy(),
                        dropShadow->stdDeviationX(), dropShadow->stdDeviationY(),
                        dropShadow->floodColor(), dropShadow->floodOpacity() );
                    applied = true;
                    break; // feDropShadow is a shorthand — typically only one per filter
                }
            }

            if ( !applied )
            {
                // No recognized primitives — just pop the group unchanged
                aCanvas.PopGroup();
            }
        }
    }
}
