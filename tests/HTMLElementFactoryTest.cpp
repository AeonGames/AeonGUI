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

// Phase 2 verification:
//  * Namespaced construction picks the correct HTML class.
//  * Wildcard fallback still finds SVG built-ins (with or without ns).
//  * SVG class hierarchy unchanged when no namespace is given.

#include <gtest/gtest.h>
#include "aeongui/ElementFactory.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/HTMLElement.hpp"
#include "aeongui/dom/HTMLDivElement.hpp"
#include "aeongui/dom/HTMLBodyElement.hpp"
#include "aeongui/dom/HTMLImageElement.hpp"
#include "aeongui/dom/SVGElement.hpp"
#include "aeongui/dom/SVGRectElement.hpp"
#include "aeongui/dom/SVGSVGElement.hpp"

namespace
{
    constexpr const char* kXHTML = "http://www.w3.org/1999/xhtml";
    constexpr const char* kSVG   = "http://www.w3.org/2000/svg";
}

TEST ( HTMLElementFactory, ConstructsHTMLDivInXHTMLNamespace )
{
    auto elem = AeonGUI::Construct ( kXHTML, "div", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( elem, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLDivElement*> ( elem.get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLElement*> ( elem.get() ), nullptr );
}

TEST ( HTMLElementFactory, ConstructsHTMLBodyInXHTMLNamespace )
{
    auto elem = AeonGUI::Construct ( kXHTML, "body", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( elem, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLBodyElement*> ( elem.get() ), nullptr );
}

TEST ( HTMLElementFactory, ConstructsHTMLImageInXHTMLNamespace )
{
    auto elem = AeonGUI::Construct ( kXHTML, "img", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( elem, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLImageElement*> ( elem.get() ), nullptr );
}

TEST ( HTMLElementFactory, InlineSVGInsideHTMLFallsBackToSVG )
{
    // <svg> inside an XHTML document carries the SVG namespace
    // (libxml2 supplies element->ns->href). Even if it didn't, the
    // wildcard fallback to the SVG built-ins should still match.
    auto svg = AeonGUI::Construct ( kSVG, "svg", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( svg, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::SVGSVGElement*> ( svg.get() ), nullptr );

    auto rect = AeonGUI::Construct ( kSVG, "rect", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( rect, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::SVGRectElement*> ( rect.get() ), nullptr );
}

TEST ( HTMLElementFactory, BareNameStillResolvesToSVGBuiltin )
{
    // Existing API: Construct(name, ...) with no namespace must keep working
    // for SVG content that doesn't carry an xmlns attribute.
    auto rect = AeonGUI::Construct ( "rect", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( rect, nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::SVGRectElement*> ( rect.get() ), nullptr );
}

TEST ( HTMLElementFactory, UnknownXHTMLTagFallsBackToGenericElement )
{
    // XHTML tag we haven't registered (e.g. "table" — not in v1 set):
    // wildcard fallback won't find it either, so factory returns a
    // generic Element, NOT a wrong-typed SVG class.
    auto elem = AeonGUI::Construct ( kXHTML, "table", AeonGUI::AttributeMap{}, nullptr );
    ASSERT_NE ( elem, nullptr );
    // Must NOT be any HTML or SVG concrete subclass.
    EXPECT_EQ ( dynamic_cast<AeonGUI::DOM::HTMLElement*> ( elem.get() ), nullptr );
    EXPECT_EQ ( dynamic_cast<AeonGUI::DOM::SVGElement*> ( elem.get() ), nullptr );
}
