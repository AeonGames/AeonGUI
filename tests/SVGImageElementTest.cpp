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
#include "aeongui/dom/SVGImageElement.hpp"
#ifdef AEONGUI_USE_SKIA
#include "aeongui/SkiaCanvas.hpp"
namespace AeonGUI
{
    using TestCanvas = SkiaCanvas;
}
#else
#include "aeongui/CairoCanvas.hpp"
namespace AeonGUI
{
    using TestCanvas = CairoCanvas;
}
#endif

TEST ( SVGImageElementTest, ParsesCoreAttributes )
{
    AeonGUI::AttributeMap attributes
    {
        {"x", "10"},
        {"y", "20"},
        {"width", "30"},
        {"height", "40"},
        {"href", "images/example.pcx"},
        {"preserveAspectRatio", "xMinYMax slice"},
        {"crossorigin", "anonymous"},
        {"decoding", "sync"}
    };

    AeonGUI::DOM::SVGImageElement image{"image", std::move ( attributes ), nullptr};

    EXPECT_FLOAT_EQ ( image.x().baseVal().value(), 10.0f );
    EXPECT_FLOAT_EQ ( image.y().baseVal().value(), 20.0f );
    EXPECT_FLOAT_EQ ( image.width().baseVal().value(), 30.0f );
    EXPECT_FLOAT_EQ ( image.height().baseVal().value(), 40.0f );
    EXPECT_EQ ( image.href().baseVal(), "images/example.pcx" );
    EXPECT_EQ ( image.preserveAspectRatio().baseVal().GetAlign(), AeonGUI::PreserveAspectRatio::Align::XMinYMax );
    EXPECT_EQ ( image.preserveAspectRatio().baseVal().GetMeetOrSlice(), AeonGUI::PreserveAspectRatio::MeetOrSlice::Slice );
    EXPECT_EQ ( image.crossOrigin(), "anonymous" );
    EXPECT_EQ ( image.decoding(), "sync" );
}

TEST ( SVGImageElementTest, SupportsXlinkHrefFallback )
{
    AeonGUI::AttributeMap attributes
    {
        {"xlink:href", "images/fallback.pcx"}
    };

    AeonGUI::DOM::SVGImageElement image{"image", std::move ( attributes ), nullptr};
    EXPECT_EQ ( image.href().baseVal(), "images/fallback.pcx" );
}

TEST ( SVGImageElementTest, DrawWithMissingHrefDoesNotThrow )
{
    AeonGUI::AttributeMap attributes
    {
        {"width", "10"},
        {"height", "10"}
    };
    AeonGUI::DOM::SVGImageElement image{"image", std::move ( attributes ), nullptr};
    AeonGUI::TestCanvas canvas{32, 32};

    EXPECT_NO_THROW ( image.DrawStart ( canvas ) );
}
