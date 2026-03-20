/*
Copyright (C) 2024-2026 Rodrigo Jose Hernandez Cordoba

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
#include <cstddef>
#include "aeongui/Platform.hpp"
namespace AeonGUI
{
    /** @brief SVG viewBox attribute value.
     *
     *  Stores the min-x, min-y, width, and height defining the SVG viewport rectangle.
     */
    struct ViewBox
    {
        double min_x{};   ///< Minimum X coordinate of the viewport.
        double min_y{};   ///< Minimum Y coordinate of the viewport.
        double width{};   ///< Width of the viewport.
        double height{};  ///< Height of the viewport.
    };

    /** @brief SVG preserveAspectRatio attribute.
     *
     *  Controls how an SVG element with a viewBox is scaled and aligned
     *  within its viewport. See the SVG specification for details.
     */
    class DLL PreserveAspectRatio
    {
    public:
        /** @brief Default constructor. Initializes to xMidYMid meet. */
        PreserveAspectRatio() = default;
        /** @brief Construct from a string representation.
         *  @param aString A string such as "xMidYMid meet".
         */
        PreserveAspectRatio ( std::string_view aString );
        /** @brief Axis alignment selector values. */
        enum MinMidMax : uint8_t
        {
            Min = 0x1, ///< Align to the minimum (start) edge.
            Mid = 0x2, ///< Align to the midpoint.
            Max = 0x3  ///< Align to the maximum (end) edge.
        };
        /** @brief Combined X and Y alignment values for preserveAspectRatio. */
        enum Align : uint8_t
        {
            none = 0,                      ///< Do not force uniform scaling.
            XMinYMin = Min << 4 | Min,     ///< Align X-min, Y-min.
            XMinYMid = Min << 4 | Mid,     ///< Align X-min, Y-mid.
            XMinYMax = Min << 4 | Max,     ///< Align X-min, Y-max.
            XMidYMin = Mid << 4 | Min,     ///< Align X-mid, Y-min.
            XMidYMid = Mid << 4 | Mid,     ///< Align X-mid, Y-mid (default).
            XMidYMax = Mid << 4 | Max,     ///< Align X-mid, Y-max.
            XMaxYMin = Max << 4 | Min,     ///< Align X-max, Y-min.
            XMaxYMid = Max << 4 | Mid,     ///< Align X-max, Y-mid.
            XMaxYMax = Max << 4 | Max      ///< Align X-max, Y-max.
        };
        /** @brief Meet-or-slice scaling strategy. */
        enum class MeetOrSlice
        {
            Meet,  ///< Scale to fit entirely within the viewport.
            Slice  ///< Scale to cover the viewport, clipping overflow.
        };
        /** @brief Get the alignment value.
         *  @return The combined Align enumeration value.
         */
        Align GetAlign() const;
        /** @brief Get the X-axis alignment component.
         *  @return Min, Mid, or Max for the X axis.
         */
        MinMidMax GetAlignX() const;
        /** @brief Get the Y-axis alignment component.
         *  @return Min, Mid, or Max for the Y axis.
         */
        MinMidMax GetAlignY() const;
        /** @brief Get the meet-or-slice strategy.
         *  @return Meet or Slice.
         */
        MeetOrSlice GetMeetOrSlice() const;
    private:
        Align mAlign{Align::XMidYMid};
        MeetOrSlice mMeetOrSlice{MeetOrSlice::Meet};
    };

    /** @brief Convert a string to a value of type T.
     *  @tparam T The target type.
     *  @param aString The source string.
     *  @return The parsed value.
     */
    template<typename T> T FromString ( const std::string_view aString )
    {
        return T{aString};
    }

    /** @brief Convert a named attribute from an attribute map to a value of type T.
     *  @tparam T The target type.
     *  @param aAttribute The attribute name to look up.
     *  @param aAttributes The attribute map to search.
     *  @return The parsed value, or default-constructed T if not found.
     */
    template<typename T> T FromString ( const char *aAttribute, const std::unordered_map<std::string, std::string>& aAttributes )
    {
        return T{FromString<T> ( aAttributes.find ( aAttribute ) != aAttributes.end() ? aAttributes.at ( aAttribute ) : "" ) };
    }

    /** @brief Specialization of FromString for double.
     *  @param aString The source string.
     *  @return The parsed double value.
     */
    template<> double FromString<double> ( const std::string_view aString );
}
#endif
