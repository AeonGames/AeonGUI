/*
Copyright (C) 2024 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_ATTRIBUTE_H
#define AEONGUI_ATTRIBUTE_H
#include <string>
#include <unordered_map>
#include <variant>
#include <cstdint>
namespace AeonGUI
{
    struct ViewBox
    {
        double min_x{};
        double min_y{};
        double width{};
        double height{};
    };

    class PreserveAspectRatio
    {
    public:
        PreserveAspectRatio() = default;
        PreserveAspectRatio ( std::string_view aString );
        enum class Align
        {
            None,
            XMinYMin,
            XMidYMin,
            XMaxYMin,
            XMinYMid,
            XMidYMid,
            XMaxYMid,
            XMinYMax,
            XMidYMax,
            XMaxYMax
        };
        enum class MeetOrSlice
        {
            Meet,
            Slice
        };
    private:
        Align mAlign{Align::XMidYMid};
        MeetOrSlice mMeetOrSlice{MeetOrSlice::Meet};
    };

    template<typename T> T FromString ( const std::string_view aString )
    {
        return T{aString};
    }

    template<typename T> T FromString ( const char *aAttribute, const std::unordered_map<std::string, std::string>& aAttributes )
    {
        return T{FromString<T> ( aAttributes.find ( aAttribute ) != aAttributes.end() ? aAttributes.at ( aAttribute ) : "" ) };
    }

    template<> double FromString<double> ( const std::string_view aString );
}
#endif
