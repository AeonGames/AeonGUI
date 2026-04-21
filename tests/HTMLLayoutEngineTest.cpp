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

// Phase 3 verification:
//  * HTMLLayoutEngine builds a Yoga tree mirroring the HTMLElement
//    subtree, reads width/height/margin/padding/flex-direction/display
//    from libcss computed styles, and writes the resulting boxes back
//    onto each element in document coordinates.
//  * Non-HTML DOM children (text nodes, etc.) are skipped — they don't
//    participate in flex layout in this slice.

#include <gtest/gtest.h>
#include <memory>

#include "aeongui/ElementFactory.hpp"
#include "aeongui/HTMLLayoutEngine.hpp"
#include "aeongui/dom/HTMLElement.hpp"
#include "aeongui/dom/HTMLHtmlElement.hpp"
#include "aeongui/dom/HTMLBodyElement.hpp"
#include "aeongui/dom/HTMLDivElement.hpp"

namespace
{
    constexpr const char* kXHTML = "http://www.w3.org/1999/xhtml";

    /// Construct an HTML element via the factory and downcast.  Asserts
    /// rather than returning nullptr so test failures point at the
    /// offending construction site.
    template<class T>
    std::unique_ptr<T> MakeHTMLElement ( const char* aTag, const char* aStyle, AeonGUI::DOM::Node* aParent )
    {
        AeonGUI::AttributeMap attrs;
        if ( aStyle && aStyle[0] != '\0' )
        {
            attrs.emplace ( "style", aStyle );
        }
        auto elem = AeonGUI::Construct ( kXHTML, aTag, std::move ( attrs ), aParent );
        T* typed = dynamic_cast<T*> ( elem.get() );
        EXPECT_NE ( typed, nullptr ) << "factory did not produce expected type for <" << aTag << ">";
        // Transfer ownership through the typed pointer.
        elem.release();
        return std::unique_ptr<T> ( typed );
    }

    /// Helper to attach an element to a parent and return a raw pointer.
    template<class T>
    T* Attach ( AeonGUI::DOM::Node* aParent, std::unique_ptr<T> aChild )
    {
        T* raw = aChild.get();
        aParent->AddNode ( std::move ( aChild ) );
        return raw;
    }
}

TEST ( HTMLLayoutEngine, RootGetsViewportSize )
{
    auto html = MakeHTMLElement<AeonGUI::DOM::HTMLHtmlElement> ( "html", "", nullptr );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( html.get(), 800.0f, 600.0f );

    const auto& box = html->GetLayoutBox();
    EXPECT_FLOAT_EQ ( box.x,      0.0f );
    EXPECT_FLOAT_EQ ( box.y,      0.0f );
    EXPECT_FLOAT_EQ ( box.width,  800.0f );
    EXPECT_FLOAT_EQ ( box.height, 600.0f );
}

TEST ( HTMLLayoutEngine, ChildrenStackInDocumentFlow )
{
    // Default flex-direction is column → children stack vertically.
    auto body = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> ( "body", "", nullptr );
    auto* a = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "height: 40px", body.get() ) );
    auto* b = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "height: 60px", body.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 200.0f, 300.0f );

    EXPECT_FLOAT_EQ ( a->GetLayoutBox().y,      0.0f );
    EXPECT_FLOAT_EQ ( a->GetLayoutBox().height, 40.0f );
    EXPECT_FLOAT_EQ ( a->GetLayoutBox().width,  200.0f );  // stretches to container

    EXPECT_FLOAT_EQ ( b->GetLayoutBox().y,      40.0f );
    EXPECT_FLOAT_EQ ( b->GetLayoutBox().height, 60.0f );
    EXPECT_FLOAT_EQ ( b->GetLayoutBox().width,  200.0f );
}

TEST ( HTMLLayoutEngine, FlexDirectionRowLaysChildrenHorizontally )
{
    auto body = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                    "body", "display: flex; flex-direction: row", nullptr );
    auto* a = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "width: 80px; height: 50px", body.get() ) );
    auto* b = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "width: 120px; height: 50px", body.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 400.0f, 200.0f );

    EXPECT_FLOAT_EQ ( a->GetLayoutBox().x,     0.0f );
    EXPECT_FLOAT_EQ ( a->GetLayoutBox().width, 80.0f );

    EXPECT_FLOAT_EQ ( b->GetLayoutBox().x,     80.0f );
    EXPECT_FLOAT_EQ ( b->GetLayoutBox().width, 120.0f );
}

TEST ( HTMLLayoutEngine, PaddingShrinksContentArea )
{
    // 200x100 container with 10px padding on every side.  The single
    // child fills the inner content area and should sit at (10, 10)
    // with size 180 x 80 in document coordinates.
    auto outer = MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                     "div", "width: 200px; height: 100px; padding: 10px", nullptr );
    auto* inner = Attach ( outer.get(),
                           MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "", outer.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( outer.get(), 800.0f, 600.0f );

    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().x,      10.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().y,      10.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().width,  180.0f );
}

TEST ( HTMLLayoutEngine, MarginOffsetsChildBox )
{
    auto outer = MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                     "div", "width: 200px; height: 100px", nullptr );
    auto* inner = Attach ( outer.get(),
                           MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                               "div", "width: 50px; height: 30px; margin: 5px", outer.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( outer.get(), 800.0f, 600.0f );

    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().x,      5.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().y,      5.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().width,  50.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().height, 30.0f );
}

TEST ( HTMLLayoutEngine, DisplayNoneCollapsesToZero )
{
    auto body = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> ( "body", "", nullptr );
    auto* a = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "height: 40px", body.get() ) );
    auto* hidden = Attach ( body.get(),
                            MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                                "div", "height: 100px; display: none", body.get() ) );
    auto* c = Attach ( body.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> ( "div", "height: 60px", body.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 200.0f, 300.0f );

    EXPECT_FLOAT_EQ ( a->GetLayoutBox().y,      0.0f );
    EXPECT_FLOAT_EQ ( a->GetLayoutBox().height, 40.0f );

    // display: none collapses to a 0x0 box and contributes no flow.
    EXPECT_FLOAT_EQ ( hidden->GetLayoutBox().width,  0.0f );
    EXPECT_FLOAT_EQ ( hidden->GetLayoutBox().height, 0.0f );

    // The next sibling sits directly below `a`, not below `hidden`.
    EXPECT_FLOAT_EQ ( c->GetLayoutBox().y,      40.0f );
    EXPECT_FLOAT_EQ ( c->GetLayoutBox().height, 60.0f );
}

TEST ( HTMLLayoutEngine, PercentageWidthResolvesAgainstParent )
{
    auto outer = MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                     "div", "width: 400px; height: 100px", nullptr );
    auto* inner = Attach ( outer.get(),
                           MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                               "div", "width: 50%; height: 100%", outer.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( outer.get(), 800.0f, 600.0f );

    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().width,  200.0f );
    EXPECT_FLOAT_EQ ( inner->GetLayoutBox().height, 100.0f );
}
