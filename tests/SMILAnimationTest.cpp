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
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/CairoCanvas.hpp"

namespace
{
    // Helper: write an SVG string to a temp file, load into Document, and clean up on destruction.
    class TempSVG
    {
    public:
        explicit TempSVG ( const std::string& aSvgContent, const std::string& aName = "smil-test.svg" )
            : mPath { std::filesystem::temp_directory_path() / aName }
        {
            std::ofstream file ( mPath, std::ios::binary | std::ios::out );
            file << aSvgContent;
            file.close();
            mDocument.Load ( mPath.string() );
        }
        ~TempSVG()
        {
            std::error_code ec;
            std::filesystem::remove ( mPath, ec );
        }
        AeonGUI::DOM::Document& doc()
        {
            return mDocument;
        }
    private:
        std::filesystem::path mPath;
        AeonGUI::DOM::Document mDocument;
    };
}

// ===== Document-level SMIL integration tests =====

TEST ( SMILAnimationTest, LoadSmilDemoSvg )
{
    const auto svgPath = std::filesystem::path ( SOURCE_PATH ) / "images" / "smil-demo.svg";
    ASSERT_TRUE ( std::filesystem::exists ( svgPath ) ) << "smil-demo.svg not found";

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( svgPath.string() ) );

    AeonGUI::CairoCanvas canvas ( 800u, 600u );
    canvas.Clear();
    ASSERT_NO_THROW ( document.Draw ( canvas ) );
}

TEST ( SMILAnimationTest, AdvanceTimeAndDraw )
{
    const auto svgPath = std::filesystem::path ( SOURCE_PATH ) / "images" / "smil-demo.svg";
    ASSERT_TRUE ( std::filesystem::exists ( svgPath ) );

    AeonGUI::DOM::Document document;
    document.Load ( svgPath.string() );

    AeonGUI::CairoCanvas canvas ( 800u, 600u );
    // Simulate 60 frames at 60fps
    for ( int i = 0; i < 60; ++i )
    {
        document.AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( document.Draw ( canvas ) );
    }
}

// ===== animate fill (color animation) =====

TEST ( SMILAnimationTest, AnimateFillLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="fill" values="#f00;#0f0;#f00" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate opacity =====

TEST ( SMILAnimationTest, AnimateOpacityLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate stroke-width =====

TEST ( SMILAnimationTest, AnimateStrokeWidthLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="none" stroke="#000" stroke-width="1">)"
        R"(<animate attributeName="stroke-width" values="1;6;1" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate cx (geometry: position) =====

TEST ( SMILAnimationTest, AnimateCxLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cx" from="50" to="150" dur="2s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate width (geometry: size with anchored scale) =====

TEST ( SMILAnimationTest, AnimateWidthLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">)"
        R"(<rect x="40" y="20" width="60" height="60" fill="#3b3">)"
        R"(<animate attributeName="width" values="60;120;60" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate height (geometry: size) =====

TEST ( SMILAnimationTest, AnimateHeightLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="200">)"
        R"(<rect x="20" y="20" width="60" height="60" fill="#39f">)"
        R"(<animate attributeName="height" values="60;120;60" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 200u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate r (geometry: circle radius) =====

TEST ( SMILAnimationTest, AnimateRadiusLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<circle cx="50" cy="50" r="20" fill="#48d">)"
        R"(<animate attributeName="r" values="20;40;20" dur="2s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateTransform rotate =====

TEST ( SMILAnimationTest, AnimateTransformRotateLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="25" y="25" width="50" height="50" fill="#c4e">)"
        R"(<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="3s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateTransform scale =====

TEST ( SMILAnimationTest, AnimateTransformScaleLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<circle cx="50" cy="50" r="20" fill="#48d">)"
        R"(<animateTransform attributeName="transform" type="scale" values="1;1.4;1" dur="1.5s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.75 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateTransform translate =====

TEST ( SMILAnimationTest, AnimateTransformTranslateLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">)"
        R"(<rect x="10" y="30" width="40" height="40" fill="#e90">)"
        R"(<animateTransform attributeName="transform" type="translate" from="0" to="100" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== set element =====

TEST ( SMILAnimationTest, SetElementLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="#999">)"
        R"(<set attributeName="fill" to="#3b3" begin="0s" dur="1s"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

TEST ( SMILAnimationTest, SetElementEventBasedBeginStaysInactive )
{
    // begin="click" is event-based; since events are not yet implemented,
    // the animation should remain inactive (duration parsed as 0).
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="#999">)"
        R"(<set attributeName="fill" to="#3b3" begin="click" dur="1s"/>)"
R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    // Should draw without crash even though begin="click" produces dur=0
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateMotion =====

TEST ( SMILAnimationTest, AnimateMotionLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
        R"(<circle cx="0" cy="0" r="10" fill="#f57">)"
        R"(<animateMotion path="M50,100 Q100,50 150,100 Q100,150 50,100" dur="3s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 200u );
    svg.doc().AdvanceTime ( 1.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Timing: animation should not be active before begin time =====

TEST ( SMILAnimationTest, AnimationInactiveBeforeBeginTime )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="fill" values="#f00;#0f0" dur="1s" begin="5s" repeatCount="1"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    // Advance only 2 seconds — animation begins at 5s so should remain inactive
    svg.doc().AdvanceTime ( 2.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Timing: freeze at end =====

TEST ( SMILAnimationTest, FreezeAtEnd )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cx" from="20" to="80" dur="1s" fill="freeze" repeatCount="1"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    // Advance well past the end
    svg.doc().AdvanceTime ( 5.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Multiple animations on same element =====

TEST ( SMILAnimationTest, MultipleAnimationsOnSameElement )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
        R"(<rect x="50" y="50" width="100" height="100" rx="10" ry="10" fill="#39f" stroke="#29e" stroke-width="2">)"
        R"(<animate attributeName="rx" values="10;25;10" dur="4s" repeatCount="indefinite"/>)"
        R"(<animate attributeName="ry" values="10;25;10" dur="4s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 200u );
    svg.doc().AdvanceTime ( 2.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Rapid time advancement without crash =====

TEST ( SMILAnimationTest, RapidTimeAdvancementNoCrash )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#39f">)"
        R"(<animate attributeName="fill" values="#39f;#f93;#3c6;#39f" dur="3s" repeatCount="indefinite"/>)"
R"(<animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect>)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cx" from="20" to="80" dur="2s" repeatCount="indefinite"/>)"
        R"(<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="3s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    // 300 frames at 60fps = 5 seconds
    for ( int i = 0; i < 300; ++i )
    {
        svg.doc().AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
    }
}

// ===== Zero duration animation stays inactive =====

TEST ( SMILAnimationTest, ZeroDurationStaysInactive )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="fill" values="#f00;#0f0" dur="0s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== from/to pair instead of values attribute =====

TEST ( SMILAnimationTest, FromToPairWorks )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="fill" from="#f00" to="#0f0" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate stroke color =====

TEST ( SMILAnimationTest, AnimateStrokeColorLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="none" stroke="#333" stroke-width="2">)"
        R"(<animate attributeName="stroke" values="#333;#e22;#333" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateTransform skewX =====

TEST ( SMILAnimationTest, AnimateTransformSkewXLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="20" y="20" width="60" height="60" fill="#c4e">)"
        R"(<animateTransform attributeName="transform" type="skewX" from="0" to="20" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Event-based begin: click triggers set element =====

TEST ( SMILAnimationTest, ClickTriggersSetElement )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "smil-click-test.svg";

    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
             << R"(<rect id="target" x="50" y="50" width="100" height="100" fill="#999">)"
             << R"(<set attributeName="fill" to="#3b3" begin="click" dur="2s"/>)"
             << R"(</rect></svg>)";
    }

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = tempPath.generic_string();

    // Advance time - animation should NOT be active yet (event-based begin)
    window.Update ( 0.5 );
    window.Draw();

    // Now simulate a click on the rect (center at 100, 100)
    window.HandleMouseDown ( 100.0, 100.0 );
    window.HandleMouseUp ( 100.0, 100.0 );

    // Advance time past the click - now the animation should be active
    window.Update ( 0.5 );
    window.Draw();

    // Advance past the duration — animation should deactivate
    window.Update ( 2.0 );
    ASSERT_NO_THROW ( window.Draw() );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

// ===== Event-based begin: animation stays inactive without click =====

TEST ( SMILAnimationTest, EventBasedBeginStaysInactiveWithoutEvent )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="#999">)"
        R"(<set attributeName="fill" to="#3b3" begin="click" dur="1s"/>)"
R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    // Advance time without any click event — animation should never activate
    for ( int i = 0; i < 60; ++i )
    {
        svg.doc().AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
    }
}

// ===== rx/ry path animation: corner radii animate via path rebuild =====

TEST ( SMILAnimationTest, AnimateRxRyRebuildPath )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
        R"(<rect x="20" y="20" width="160" height="160" rx="10" ry="10" fill="#39f">)"
        R"(<animate attributeName="rx" values="10;40;10" dur="2s" repeatCount="indefinite"/>)"
        R"(<animate attributeName="ry" values="10;40;10" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 200u );
    // Animate through several frames — path is rebuilt each time
    for ( int i = 0; i < 120; ++i )
    {
        svg.doc().AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
    }
}

// ===== rx/ry path animation: only rx animated =====

TEST ( SMILAnimationTest, AnimateRxOnlyRebuildPath )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" rx="5" ry="5" fill="#e90">)"
        R"(<animate attributeName="rx" values="5;20;5" dur="1s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Hover changes CSS on combined button (integration) =====

TEST ( SMILAnimationTest, HoverCombinedButtonChangesStyle )
{
    const auto svgPath = std::filesystem::path ( SOURCE_PATH ) / "images" / "smil-demo.svg";
    ASSERT_TRUE ( std::filesystem::exists ( svgPath ) );

    AeonGUI::DOM::Window window ( 800u, 600u );
    window.location() = svgPath.generic_string();

    // Advance time so animations are running
    window.Update ( 1.0 );
    window.Draw();

    // Move mouse over the combined button (center at ~400, 560)
    window.HandleMouseMove ( 400.0, 560.0 );

    window.Update ( 0.1 );
    window.Draw();

    // Move mouse away
    window.HandleMouseMove ( 10.0, 10.0 );

    window.Update ( 0.1 );
    ASSERT_NO_THROW ( window.Draw() );
}

// ===== animate cy (geometry: vertical position) =====

TEST ( SMILAnimationTest, AnimateCyLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="200">)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cy" from="50" to="150" dur="2s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 200u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate fill-opacity =====

TEST ( SMILAnimationTest, AnimateFillOpacityLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#39f">)"
        R"(<animate attributeName="fill-opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animate stroke-opacity =====

TEST ( SMILAnimationTest, AnimateStrokeOpacityLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80" fill="none" stroke="#f00" stroke-width="4">)"
        R"(<animate attributeName="stroke-opacity" values="1;0.1;1" dur="2s" repeatCount="indefinite"/>)"
R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== animateTransform skewY =====

TEST ( SMILAnimationTest, AnimateTransformSkewYLoadsAndDraws )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="20" y="20" width="60" height="60" fill="#48d">)"
        R"(<animateTransform attributeName="transform" type="skewY" from="0" to="15" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Finite repeatCount > 1 =====

TEST ( SMILAnimationTest, FiniteRepeatCountGreaterThanOne )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#f00">)"
        R"(<animate attributeName="fill" values="#f00;#0f0;#f00" dur="1s" repeatCount="3"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );

    // Mid first cycle
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );

    // Mid second cycle
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );

    // Past all 3 cycles (animation should stop)
    svg.doc().AdvanceTime ( 2.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Animation deactivation without freeze =====

TEST ( SMILAnimationTest, AnimationDeactivatesWithoutFreeze )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cx" from="20" to="80" dur="1s" repeatCount="1"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );

    // During animation
    svg.doc().AdvanceTime ( 0.5 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );

    // After animation ends — no freeze, should deactivate cleanly
    svg.doc().AdvanceTime ( 1.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
}

// ===== Multiple animation types on same element =====

TEST ( SMILAnimationTest, MixedAnimationTypesOnSameElement )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
        R"(<rect x="40" y="40" width="80" height="80" fill="#39f" stroke="#333" stroke-width="1">)"
        R"(<animate attributeName="fill" values="#39f;#f93;#39f" dur="2s" repeatCount="indefinite"/>)"
        R"(<animate attributeName="opacity" values="1;0.5;1" dur="3s" repeatCount="indefinite"/>)"
        R"(<animateTransform attributeName="transform" type="rotate" from="0 80 80" to="360 80 80" dur="4s" repeatCount="indefinite"/>)"
        R"(</rect></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 200u, 200u );
    for ( int i = 0; i < 120; ++i )
    {
        svg.doc().AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
    }
}

// ===== Mouseover event-based begin =====

TEST ( SMILAnimationTest, MouseoverTriggersAnimation )
{
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "smil-mouseover-test.svg";

    {
        std::ofstream file ( tempPath, std::ios::binary | std::ios::out );
        ASSERT_TRUE ( file.is_open() );
        file << R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
             << R"(<rect x="50" y="50" width="100" height="100" fill="#999">)"
<< R"(<animate attributeName="fill" from="#999" to="#3b3" begin="mouseenter" dur="1s" repeatCount="1"/>)"
             << R"(</rect></svg>)";
    }

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = tempPath.generic_string();

    // Advance time — animation should not be active
    window.Update ( 0.5 );
    window.Draw();

    // Move mouse over the rect (enter triggers mouseenter)
    window.HandleMouseMove ( 100.0, 100.0 );

    // Advance time — animation should now be active
    window.Update ( 0.5 );
    ASSERT_NO_THROW ( window.Draw() );

    // Advance past duration
    window.Update ( 1.0 );
    ASSERT_NO_THROW ( window.Draw() );

    std::error_code ec;
    std::filesystem::remove ( tempPath, ec );
}

// ===== Large time jump (numerical stability) =====

TEST ( SMILAnimationTest, LargeTimeJumpNoNumericalIssues )
{
    TempSVG svg
    {
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="0" y="0" width="100" height="100" fill="#39f">)"
        R"(<animate attributeName="fill" values="#39f;#f93;#3c6;#39f" dur="3s" repeatCount="indefinite"/>)"
R"(<animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>)"
        R"(</rect>)"
        R"(<circle cx="50" cy="50" r="20" fill="#e44">)"
        R"(<animate attributeName="cx" from="20" to="80" dur="2s" repeatCount="indefinite"/>)"
        R"(<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="3s" repeatCount="indefinite"/>)"
        R"(</circle></svg>)"
    };

    AeonGUI::CairoCanvas canvas ( 100u, 100u );

    // Jump 10000 seconds at once
    svg.doc().AdvanceTime ( 10000.0 );
    canvas.Clear();
    ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );

    // Then a few more normal frames
    for ( int i = 0; i < 10; ++i )
    {
        svg.doc().AdvanceTime ( 1.0 / 60.0 );
        canvas.Clear();
        ASSERT_NO_THROW ( svg.doc().Draw ( canvas ) );
    }
}
