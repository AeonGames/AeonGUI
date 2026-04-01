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

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <string>
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/EventListener.hpp"
#include "aeongui/dom/MouseEvent.hpp"

namespace
{
    /// Helper to write a temporary SVG and clean it up on destruction.
    class TempSVG
    {
    public:
        explicit TempSVG ( const std::string& aSvgContent, const std::string& aName = "aeongui-hittest.svg" )
            : mPath{std::filesystem::temp_directory_path() / aName}
        {
            std::ofstream file ( mPath, std::ios::binary | std::ios::out );
            file << aSvgContent;
        }
        ~TempSVG()
        {
            std::error_code ec;
            std::filesystem::remove ( mPath, ec );
        }
        std::string path() const
        {
            return mPath.generic_string();
        }
    private:
        std::filesystem::path mPath;
    };

    /// Listener that records calls and the event target.
    class HitTestListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            callCount++;
            lastType = event.type();
        }
        int callCount{0};
        AeonGUI::DOM::DOMString lastType;
    };
}

// ===== Basic hit test: click inside a rect ============================

TEST ( HitTestTest, HoverOverRect )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Move mouse to center of the rect
    window.HandleMouseMove ( 100.0, 100.0 );

    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_TRUE ( r1->isHover() );
}

// ===== Hit test miss: click outside any element =======================

TEST ( HitTestTest, HoverOverEmptyArea )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Move outside the rect
    window.HandleMouseMove ( 10.0, 10.0 );

    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_FALSE ( r1->isHover() );
}

// ===== Hover enters then leaves =======================================

TEST ( HitTestTest, HoverEnterAndLeave )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Enter the rect
    window.HandleMouseMove ( 100.0, 100.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_TRUE ( r1->isHover() );

    // Leave the rect — draw to rebuild pick buffer after hover ReselectCSS
    window.Draw();
    window.HandleMouseMove ( 10.0, 10.0 );
    EXPECT_FALSE ( r1->isHover() );
}

// ===== Multiple elements: topmost wins ================================

TEST ( HitTestTest, TopmostElementGetsHover )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="bottom" x="50" y="50" width="100" height="100" fill="red"/>
        <rect id="top"    x="50" y="50" width="100" height="100" fill="blue"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    window.HandleMouseMove ( 100.0, 100.0 );

    AeonGUI::DOM::Element* top = window.document()->getElementById ( "top" );
    AeonGUI::DOM::Element* bottom = window.document()->getElementById ( "bottom" );
    ASSERT_NE ( top, nullptr );
    ASSERT_NE ( bottom, nullptr );
    EXPECT_TRUE ( top->isHover() );
    EXPECT_FALSE ( bottom->isHover() );
}

// ===== Partially overlapping rects: hit each area =====================

TEST ( HitTestTest, PartialOverlapHitsCorrectElement )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="left"  x="10" y="50" width="100" height="100" fill="red"/>
        <rect id="right" x="90" y="50" width="100" height="100" fill="blue"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Hit left-only area
    window.HandleMouseMove ( 30.0, 100.0 );
    AeonGUI::DOM::Element* left = window.document()->getElementById ( "left" );
    AeonGUI::DOM::Element* right = window.document()->getElementById ( "right" );
    ASSERT_NE ( left, nullptr );
    ASSERT_NE ( right, nullptr );
    EXPECT_TRUE ( left->isHover() );
    EXPECT_FALSE ( right->isHover() );

    // Draw to rebuild pick buffer, then hit overlap area (right is on top)
    window.Draw();
    window.HandleMouseMove ( 100.0, 100.0 );
    EXPECT_FALSE ( left->isHover() );
    EXPECT_TRUE ( right->isHover() );

    // Draw to rebuild, then hit right-only area
    window.Draw();
    window.HandleMouseMove ( 170.0, 100.0 );
    EXPECT_FALSE ( left->isHover() );
    EXPECT_TRUE ( right->isHover() );
}

// ===== Circle hit test ================================================

TEST ( HitTestTest, CircleHitTest )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <circle id="c1" cx="100" cy="100" r="50" fill="green"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Hit center
    window.HandleMouseMove ( 100.0, 100.0 );
    AeonGUI::DOM::Element* c1 = window.document()->getElementById ( "c1" );
    ASSERT_NE ( c1, nullptr );
    EXPECT_TRUE ( c1->isHover() );

    // Miss at corner (outside circle radius)
    window.Draw();
    window.HandleMouseMove ( 10.0, 10.0 );
    EXPECT_FALSE ( c1->isHover() );
}

// ===== Ellipse hit test ===============================================

TEST ( HitTestTest, EllipseHitTest )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
        <ellipse id="e1" cx="100" cy="50" rx="80" ry="30" fill="purple"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 100u );
    window.location() = svg.path();
    window.Draw();

    // Center hit
    window.HandleMouseMove ( 100.0, 50.0 );
    AeonGUI::DOM::Element* e1 = window.document()->getElementById ( "e1" );
    ASSERT_NE ( e1, nullptr );
    EXPECT_TRUE ( e1->isHover() );

    // Miss above ellipse
    window.Draw();
    window.HandleMouseMove ( 100.0, 5.0 );
    EXPECT_FALSE ( e1->isHover() );
}

// ===== Group nesting: hover hits child in group =======================

TEST ( HitTestTest, GroupChildGetsHover )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <g id="group1">
            <rect id="child" x="50" y="50" width="100" height="100" fill="red"/>
        </g>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    window.HandleMouseMove ( 100.0, 100.0 );

    AeonGUI::DOM::Element* child = window.document()->getElementById ( "child" );
    ASSERT_NE ( child, nullptr );
    EXPECT_TRUE ( child->isHover() );
}

// ===== Transform: translated element =================================

TEST ( HitTestTest, TransformedElementHitTest )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="0" y="0" width="50" height="50" fill="red"
              transform="translate(100,100)"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Miss at original position (0,0 to 50,50)
    window.HandleMouseMove ( 25.0, 25.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_FALSE ( r1->isHover() );

    // Hit at translated position (100,100 to 150,150)
    window.Draw();
    window.HandleMouseMove ( 125.0, 125.0 );
    EXPECT_TRUE ( r1->isHover() );
}

// ===== ViewBox scaling: coordinates map correctly =====================

TEST ( HitTestTest, ViewBoxScaling )
{
    // viewBox is 0 0 100 100, but canvas is 200x200 → 2x scale
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect id="r1" x="0" y="0" width="50" height="50" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Viewport pixel (50,50) maps to viewBox (25,25) — inside the rect
    window.HandleMouseMove ( 50.0, 50.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_TRUE ( r1->isHover() );

    // Viewport pixel (150,150) maps to viewBox (75,75) — outside the rect
    window.Draw();
    window.HandleMouseMove ( 150.0, 150.0 );
    EXPECT_FALSE ( r1->isHover() );
}

// ===== Out of bounds: negative and beyond-viewport ====================

TEST ( HitTestTest, OutOfBoundsReturnsNoHit )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="0" y="0" width="200" height="200" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Negative coordinates
    window.HandleMouseMove ( -10.0, -10.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_FALSE ( r1->isHover() );

    // Beyond viewport
    window.HandleMouseMove ( 300.0, 300.0 );
    EXPECT_FALSE ( r1->isHover() );
}

// ===== Dirty flag: Draw returns true on first call ====================

TEST ( HitTestTest, DrawReturnsTrueOnFirstCall )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect id="r1" x="10" y="10" width="80" height="80" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 100u, 100u );
    window.location() = svg.path();

    EXPECT_TRUE ( window.Draw() );
}

// ===== Dirty flag: Draw returns false when clean ======================

TEST ( HitTestTest, DrawReturnsFalseWhenClean )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect id="r1" x="10" y="10" width="80" height="80" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 100u, 100u );
    window.location() = svg.path();

    window.Draw();
    EXPECT_FALSE ( window.Draw() );
}

// ===== Dirty flag: hover change marks dirty ===========================

TEST ( HitTestTest, HoverChangeDirties )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();   // clear dirty
    EXPECT_FALSE ( window.Draw() ); // clean

    // Hover onto rect triggers ReselectCSS → MarkDirty
    window.HandleMouseMove ( 100.0, 100.0 );
    EXPECT_TRUE ( window.Draw() );
    EXPECT_FALSE ( window.Draw() ); // clean again
}

// ===== Dirty flag: setAttribute marks dirty ===========================

TEST ( HitTestTest, SetAttributeDirties )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();
    EXPECT_FALSE ( window.Draw() );

    // setAttribute on fill triggers ReselectCSS → MarkDirty
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    r1->setAttribute ( "fill", "blue" );
    EXPECT_TRUE ( window.Draw() );
    EXPECT_FALSE ( window.Draw() );
}

// ===== Dirty flag: ResizeViewport marks dirty =========================

TEST ( HitTestTest, ResizeViewportDirties )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();
    EXPECT_FALSE ( window.Draw() );

    window.ResizeViewport ( 400u, 400u );
    EXPECT_TRUE ( window.Draw() );
    EXPECT_FALSE ( window.Draw() );
}

// ===== Mousedown sets focus on hit element ============================

TEST ( HitTestTest, MouseDownSetsFocus )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    window.HandleMouseDown ( 100.0, 100.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_TRUE ( r1->isFocus() );
    EXPECT_TRUE ( r1->isActive() );
}

// ===== Mousedown on empty clears focus ================================

TEST ( HitTestTest, MouseDownOnEmptyClearsFocus )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Focus the rect
    window.HandleMouseDown ( 100.0, 100.0 );
    window.HandleMouseUp ( 100.0, 100.0 );
    window.Draw();

    // Click empty area
    window.HandleMouseDown ( 10.0, 10.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_FALSE ( r1->isFocus() );
}

// ===== Multiple geometry types in one scene ===========================

TEST ( HitTestTest, MixedGeometryHitTest )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">
        <rect   id="r1" x="10"  y="50" width="80" height="100" fill="red"/>
        <circle id="c1" cx="200" cy="100" r="50" fill="green"/>
        <ellipse id="e1" cx="350" cy="100" rx="40" ry="60" fill="blue"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 400u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Hit rect
    window.HandleMouseMove ( 50.0, 100.0 );
    auto* r1 = window.document()->getElementById ( "r1" );
    auto* c1 = window.document()->getElementById ( "c1" );
    auto* e1 = window.document()->getElementById ( "e1" );
    ASSERT_NE ( r1, nullptr );
    ASSERT_NE ( c1, nullptr );
    ASSERT_NE ( e1, nullptr );
    EXPECT_TRUE ( r1->isHover() );
    EXPECT_FALSE ( c1->isHover() );
    EXPECT_FALSE ( e1->isHover() );

    // Hit circle
    window.Draw();
    window.HandleMouseMove ( 200.0, 100.0 );
    EXPECT_FALSE ( r1->isHover() );
    EXPECT_TRUE ( c1->isHover() );
    EXPECT_FALSE ( e1->isHover() );

    // Hit ellipse
    window.Draw();
    window.HandleMouseMove ( 350.0, 100.0 );
    EXPECT_FALSE ( r1->isHover() );
    EXPECT_FALSE ( c1->isHover() );
    EXPECT_TRUE ( e1->isHover() );
}

// ===== Mouseenter event fires on hover ================================

TEST ( HitTestTest, MouseEnterEventFires )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    HitTestListener listener;
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    r1->addEventListener ( "mouseenter", &listener );

    window.HandleMouseMove ( 100.0, 100.0 );
    EXPECT_EQ ( listener.callCount, 1 );
    EXPECT_EQ ( listener.lastType, "mouseenter" );
}

// ===== Mouseleave event fires on leave ================================

TEST ( HitTestTest, MouseLeaveEventFires )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    HitTestListener listener;
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    r1->addEventListener ( "mouseleave", &listener );

    // Enter first
    window.HandleMouseMove ( 100.0, 100.0 );
    EXPECT_EQ ( listener.callCount, 0 ); // no leave yet

    // Leave
    window.Draw();
    window.HandleMouseMove ( 10.0, 10.0 );
    EXPECT_EQ ( listener.callCount, 1 );
    EXPECT_EQ ( listener.lastType, "mouseleave" );
}

// ===== Pick buffer survives without any geometry ======================

TEST ( HitTestTest, EmptySvgNoGeometry )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 100u, 100u );
    window.location() = svg.path();
    window.Draw();

    // Should not crash — no elements to hover
    window.HandleMouseMove ( 50.0, 50.0 );
    window.HandleMouseDown ( 50.0, 50.0 );
    window.HandleMouseUp ( 50.0, 50.0 );
}

// ===== Pick buffer handles text (non-geometry) correctly ==============

TEST ( HitTestTest, TextElementNotHittable )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <text id="t1" x="100" y="100">Hello</text>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Text is not a geometry element, so it shouldn't be in the pick buffer
    window.HandleMouseMove ( 100.0, 100.0 );
    AeonGUI::DOM::Element* t1 = window.document()->getElementById ( "t1" );
    ASSERT_NE ( t1, nullptr );
    EXPECT_FALSE ( t1->isHover() );
}

// ===== Group transform applied to children ============================

TEST ( HitTestTest, GroupTransformAffectsChildHit )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <g transform="translate(100,100)">
            <rect id="r1" x="0" y="0" width="50" height="50" fill="red"/>
        </g>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    // Miss at original position
    window.HandleMouseMove ( 25.0, 25.0 );
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    EXPECT_FALSE ( r1->isHover() );

    // Hit at translated position
    window.Draw();
    window.HandleMouseMove ( 125.0, 125.0 );
    EXPECT_TRUE ( r1->isHover() );
}

// ===== Mouse move within same element does not re-fire enter ==========

TEST ( HitTestTest, MoveWithinElementNoReenter )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="r1" x="50" y="50" width="100" height="100" fill="red"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    HitTestListener enterListener;
    AeonGUI::DOM::Element* r1 = window.document()->getElementById ( "r1" );
    ASSERT_NE ( r1, nullptr );
    r1->addEventListener ( "mouseenter", &enterListener );

    window.HandleMouseMove ( 80.0, 80.0 );
    EXPECT_EQ ( enterListener.callCount, 1 );

    // Move within the same element — should not fire again
    window.Draw();
    window.HandleMouseMove ( 120.0, 120.0 );
    EXPECT_EQ ( enterListener.callCount, 1 );
}

// ===== Hover transitions between adjacent elements ====================

TEST ( HitTestTest, HoverTransitionBetweenElements )
{
    TempSVG svg{R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
        <rect id="left"  x="0"   y="50" width="100" height="100" fill="red"/>
        <rect id="right" x="100" y="50" width="100" height="100" fill="blue"/>
    </svg>)SVG"};

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = svg.path();
    window.Draw();

    HitTestListener leftLeave, rightEnter;
    auto* left = window.document()->getElementById ( "left" );
    auto* right = window.document()->getElementById ( "right" );
    ASSERT_NE ( left, nullptr );
    ASSERT_NE ( right, nullptr );
    left->addEventListener ( "mouseleave", &leftLeave );
    right->addEventListener ( "mouseenter", &rightEnter );

    // Hover left
    window.HandleMouseMove ( 50.0, 100.0 );
    EXPECT_TRUE ( left->isHover() );
    EXPECT_FALSE ( right->isHover() );

    // Move to right
    window.Draw();
    window.HandleMouseMove ( 150.0, 100.0 );
    EXPECT_FALSE ( left->isHover() );
    EXPECT_TRUE ( right->isHover() );
    EXPECT_EQ ( leftLeave.callCount, 1 );
    EXPECT_EQ ( rightEnter.callCount, 1 );
}
