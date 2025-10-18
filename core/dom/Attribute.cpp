/*
Copyright (C) 2024,2025 Rodrigo Jose Hernandez Cordoba

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

#include "aeongui/Attribute.hpp"
#include <regex>
#include <unordered_map>
namespace AeonGUI
{
    template<> double FromString<double> ( const std::string_view aString )
    {
        return std::stod ( aString.data() );
    }

    const std::regex PreserveAspectRatioRegex{"\\s*(none|xMinYMin|xMidYMin|xMaxYMin|xMinYMid|xMidYMid|xMaxYMid|xMinYMax|xMidYMax|xMaxYMax)\\s*(meet|slice)?\\s*"};
    static const std::unordered_map<std::string_view, PreserveAspectRatio::Align> PreserveAspectRatioMap
    {
        {"none", PreserveAspectRatio::Align::none},
        {"xMinYMin", PreserveAspectRatio::Align::XMinYMin},
        {"xMidYMin", PreserveAspectRatio::Align::XMidYMin},
        {"xMaxYMin", PreserveAspectRatio::Align::XMaxYMin},
        {"xMinYMid", PreserveAspectRatio::Align::XMinYMid},
        {"xMidYMid", PreserveAspectRatio::Align::XMidYMid},
        {"xMaxYMid", PreserveAspectRatio::Align::XMaxYMid},
        {"xMinYMax", PreserveAspectRatio::Align::XMinYMax},
        {"xMidYMax", PreserveAspectRatio::Align::XMidYMax},
        {"xMaxYMax", PreserveAspectRatio::Align::XMaxYMax},
    };

    PreserveAspectRatio::PreserveAspectRatio ( std::string_view aString ) : mAlign{Align::XMidYMid}, mMeetOrSlice{MeetOrSlice::Meet}
    {
        std::cmatch m;
        if ( std::regex_match ( aString.data(), m, PreserveAspectRatioRegex ) )
        {
            mAlign = PreserveAspectRatioMap.at ( m[1].str() );
            if ( m[2] == "meet" )
            {
                mMeetOrSlice = MeetOrSlice::Meet;
            }
            else if ( m[2] == "slice" )
            {
                mMeetOrSlice = MeetOrSlice::Slice;
            }
        }
    }

    PreserveAspectRatio::Align PreserveAspectRatio::GetAlign() const
    {
        return mAlign;
    }

    PreserveAspectRatio::MinMidMax PreserveAspectRatio::GetAlignX() const
    {
        return static_cast<MinMidMax> ( mAlign & 0x3 );
    }

    PreserveAspectRatio::MinMidMax PreserveAspectRatio::GetAlignY() const
    {
        return static_cast<MinMidMax> ( mAlign >> 4 & 0x3 );
    }

    PreserveAspectRatio::MeetOrSlice PreserveAspectRatio::GetMeetOrSlice() const
    {
        return mMeetOrSlice;
    }
}
