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
#include "aeongui/dom/SVGAnimatedLength.hpp"
#include "aeongui/dom/SVGAnimatedString.hpp"
#include "aeongui/dom/SVGAnimatedPreserveAspectRatio.hpp"

using namespace AeonGUI::DOM;

TEST ( SVGAnimatedTypesTest, AnimatedLengthMutableAccessors )
{
    SVGAnimatedLength length;
    length.baseVal().newValueSpecifiedUnits ( SVGLengthType::PX, 42.0f );
    length.animVal().newValueSpecifiedUnits ( SVGLengthType::PX, 84.0f );

    EXPECT_EQ ( length.baseVal().unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.baseVal().valueInSpecifiedUnits(), 42.0f );
    EXPECT_FLOAT_EQ ( length.animVal().valueInSpecifiedUnits(), 84.0f );
}

TEST ( SVGAnimatedTypesTest, AnimatedStringReadWrite )
{
    SVGAnimatedString value;
    value.baseVal() = "foo.png";
    value.animVal() = "bar.png";

    EXPECT_EQ ( value.baseVal(), "foo.png" );
    EXPECT_EQ ( value.animVal(), "bar.png" );
}

TEST ( SVGAnimatedTypesTest, AnimatedPreserveAspectRatioReadWrite )
{
    SVGAnimatedPreserveAspectRatio par;
    par.baseVal() = AeonGUI::PreserveAspectRatio{"xMinYMax slice"};
    par.animVal() = AeonGUI::PreserveAspectRatio{"none"};

    EXPECT_EQ ( par.baseVal().GetAlign(), AeonGUI::PreserveAspectRatio::Align::XMinYMax );
    EXPECT_EQ ( par.baseVal().GetMeetOrSlice(), AeonGUI::PreserveAspectRatio::MeetOrSlice::Slice );
    EXPECT_EQ ( par.animVal().GetAlign(), AeonGUI::PreserveAspectRatio::Align::none );
}
