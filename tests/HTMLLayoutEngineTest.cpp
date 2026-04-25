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
#include "aeongui/dom/Text.hpp"

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

TEST ( HTMLLayoutEngine, TextLeafGainsIntrinsicSize )
{
    // A div with no width/height set, holding only a non-trivial text
    // run.  The Yoga measure callback should report the Pango-measured
    // dimensions, so the laid-out box must be wider than 0 and taller
    // than 0 — and a clearly longer text must produce a wider box than
    // a shorter one with the same font settings.
    //
    // Wrapping in a <body> matters: the engine forces the *root*
    // element to the supplied viewport size when no explicit
    // width/height is set, which would mask the intrinsic measure.
    auto body_a = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                      "body", "display: flex; flex-direction: row", nullptr );
    auto* short_div = Attach ( body_a.get(),
                               MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                                   "div", "font-size: 16px", body_a.get() ) );
    short_div->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "Hi", short_div ) );

    auto body_b = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                      "body", "display: flex; flex-direction: row", nullptr );
    auto* long_div = Attach ( body_b.get(),
                              MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                                  "div", "font-size: 16px", body_b.get() ) );
    long_div->AddNode (
        std::make_unique<AeonGUI::DOM::Text> ( "Hello, world!", long_div ) );

    AeonGUI::HTMLLayoutEngine engine_a;
    engine_a.Layout ( body_a.get(), 800.0f, 600.0f );
    AeonGUI::HTMLLayoutEngine engine_b;
    engine_b.Layout ( body_b.get(), 800.0f, 600.0f );

    const auto& sb = short_div->GetLayoutBox();
    const auto& lb = long_div->GetLayoutBox();

    EXPECT_GT ( sb.width,  0.0f ) << "text leaf must report a positive width";
    EXPECT_GT ( sb.height, 0.0f ) << "text leaf must report a positive height";
    EXPECT_GT ( lb.width,  sb.width )
            << "longer text run should measure wider than a shorter one";
    EXPECT_FLOAT_EQ ( sb.height, lb.height )
            << "single-line runs at the same font size should be the same height";
}

TEST ( HTMLLayoutEngine, WhitespaceOnlyTextLeafCollapses )
{
    // libxml2 keeps the indentation between sibling tags as Text nodes.
    // Such nodes must not give a parent any intrinsic size; in this
    // row-flex container the whitespace-only div should report a 0
    // main-axis (width) extent, while a sibling holding real text
    // should report a positive width.
    auto body = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                    "body", "display: flex; flex-direction: row", nullptr );
    auto* ws = Attach ( body.get(),
                        MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                            "div", "font-size: 16px", body.get() ) );
    ws->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "   \n\t  ", ws ) );

    auto* real = Attach ( body.get(),
                          MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                              "div", "font-size: 16px", body.get() ) );
    real->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "abc", real ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 800.0f, 600.0f );

    EXPECT_FLOAT_EQ ( ws->GetLayoutBox().width,  0.0f )
            << "whitespace-only text run should not give the box any width";
    EXPECT_GT ( real->GetLayoutBox().width, 0.0f )
            << "sibling with real text should still measure positive width";
}

TEST ( HTMLLayoutEngine, TextWrapsAtAvailableWidth )
{
    // A long text run inside a narrow container forces the measure
    // callback to receive an AT_MOST width hint, which must make Pango
    // wrap the text into multiple lines.  Hard-coded pixel thresholds
    // are unreliable across hosts because the actual glyph metrics
    // depend on whichever font fontconfig resolves to (CI distros ship
    // very different default fonts).  Instead, lay the *same* text out
    // first in an unconstrained container to obtain a single-line
    // baseline height, then assert the constrained layout is both
    // clamped horizontally and meaningfully taller than that baseline.
    const char* kLongText =
        "The quick brown fox jumps over the lazy dog repeatedly.";
    constexpr float kContainerWidth = 80.0f;
    constexpr float kEpsilon = 0.5f;  // sub-pixel rounding tolerance

    // --- Baseline: unconstrained, expected to be a single line ---
    auto baselineBody = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                            "body", "", nullptr );
    auto* baselineDiv = Attach ( baselineBody.get(),
                                 MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                                     "div", "font-size: 16px", baselineBody.get() ) );
    baselineDiv->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( kLongText, baselineDiv ) );

    AeonGUI::HTMLLayoutEngine baselineEngine;
    // Give plenty of room so the text never wraps.
    baselineEngine.Layout ( baselineBody.get(), 100000.0f, 600.0f );
    const auto baselineBox = baselineDiv->GetLayoutBox();

    if ( baselineBox.width <= kContainerWidth || baselineBox.height <= 0.0f )
    {
        // No font available (fontconfig fallback produced zero-width
        // glyphs) or the font is so narrow the whole string already
        // fits in 80 px.  In either case the wrapping behaviour we
        // want to verify cannot be exercised meaningfully on this host.
        GTEST_SKIP() << "Pango produced an unwrappable measurement on this "
                        "host (baseline width=" << baselineBox.width
                     << ", height=" << baselineBox.height
                     << "); skipping wrap check.";
    }

    // --- Constrained: same text inside a narrow container ---
    auto body = MakeHTMLElement<AeonGUI::DOM::HTMLBodyElement> (
                    "body", "", nullptr );
    auto* div = Attach ( body.get(),
                         MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                             "div", "width: 80px; font-size: 16px", body.get() ) );
    div->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( kLongText, div ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 1000.0f, 600.0f );
    const auto box = div->GetLayoutBox();

    EXPECT_LE ( box.width, kContainerWidth + kEpsilon )
            << "wrapped text should respect the container's available width";
    // Two lines should be at least ~1.5x a single line's height once
    // line gap and leading are accounted for; this avoids depending on
    // the exact font metrics while still proving wrapping occurred.
    EXPECT_GT ( box.height, baselineBox.height * 1.5f )
            << "wrapped text should be visibly taller than a single line "
       "(baseline height=" << baselineBox.height
            << ", wrapped height=" << box.height << ")";
}

TEST ( HTMLLayoutEngine, UnknownXHTMLBlockElementStacksChildrenVertically )
{
    // <section> isn't a registered HTML built-in, so it falls back to
    // the generic HTMLElement.  The HTML UA stylesheet maps it to
    // `display: block`, so two child <div>s must stack vertically
    // (Yoga column flow), proving the UA cascade is wired up: without
    // it, libcss would compute `display: inline` and the engine would
    // have to lean on its inline-as-block safety net rather than on a
    // real CSS rule.
    auto section = MakeHTMLElement<AeonGUI::DOM::HTMLElement> (
                       "section", "", nullptr );
    auto* a = Attach ( section.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                           "div", "height: 30px", section.get() ) );
    auto* b = Attach ( section.get(),
                       MakeHTMLElement<AeonGUI::DOM::HTMLDivElement> (
                           "div", "height: 50px", section.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( section.get(), 200.0f, 400.0f );

    EXPECT_FLOAT_EQ ( a->GetLayoutBox().y,      0.0f );
    EXPECT_FLOAT_EQ ( a->GetLayoutBox().height, 30.0f );
    EXPECT_FLOAT_EQ ( b->GetLayoutBox().y,      30.0f );
    EXPECT_FLOAT_EQ ( b->GetLayoutBox().height, 50.0f );
}
