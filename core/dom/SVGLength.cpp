/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/SVGLength.hpp"
#include <unordered_map>
#include <regex>
#include <string_view>
#include <format>

namespace AeonGUI
{
    namespace DOM
    {
        const std::unordered_map<SVGLengthType, const char*> SVGLengthTypeToSuffix
        {
            {SVGLengthType::NUMBER, ""},
            {SVGLengthType::PX, "px"},
            {SVGLengthType::CM, "cm"},
            {SVGLengthType::MM, "mm"},
            {SVGLengthType::IN, "in"},
            {SVGLengthType::PT, "pt"},
            {SVGLengthType::PC, "pc"},
            {SVGLengthType::PERCENTAGE, "%"},
            {SVGLengthType::EMS, "em"},
            {SVGLengthType::EXS, "ex"}
        };

        const std::unordered_map<std::string_view, SVGLengthType> SuffixToSVGLengthType
        {
            {"", SVGLengthType::NUMBER},
            {"px", SVGLengthType::PX},
            {"cm", SVGLengthType::CM},
            {"mm", SVGLengthType::MM},
            {"in", SVGLengthType::IN},
            {"pt", SVGLengthType::PT},
            {"pc", SVGLengthType::PC},
            {"%", SVGLengthType::PERCENTAGE},
            {"em", SVGLengthType::EMS},
            {"ex", SVGLengthType::EXS}
        };

        const char* StringValueRegex{R"(([-+]?[0-9]*\.?[0-9]+)(px|cm|mm|in|pt|pc|%|em|ex)?)"};

        SVGLength::SVGLength() = default;

        SVGLength::~SVGLength() = default;

        SVGLengthType SVGLength::unitType() const
        {
            return mUnitType;
        }

        float SVGLength::value() const
        {
            return mValue;
        }

        float SVGLength::value ( float value )
        {
            mValue = value;
            return mValue;
        }

        const DOMString& SVGLength::valueAsString() const
        {
            return mValueAsString;
        }

        const DOMString& SVGLength::valueAsString ( const DOMString& value )
        {
            std::smatch match;
            if ( !std::regex_match ( value, match, std::regex ( StringValueRegex ) ) )
            {
                throw std::invalid_argument ( "Invalid SVG length string format" );
            }
            // Extract the numeric part and the unit part
            newValueSpecifiedUnits ( SuffixToSVGLengthType.at ( match[2].str().c_str() ), std::stof ( match[1].str() ) );
            return mValueAsString;
        }

        void SVGLength::UpdateValueAsString()
        {
            mValueAsString = std::format ( "{:.15g}{}", mValueInSpecifiedUnits, SVGLengthTypeToSuffix.at ( mUnitType ) );
        }

        float SVGLength::valueInSpecifiedUnits() const
        {
            return mValueInSpecifiedUnits;
        }

        float SVGLength::valueInSpecifiedUnits ( float value )
        {
            mValueInSpecifiedUnits = value;
            return mValueInSpecifiedUnits;
        }

        void SVGLength::convertToSpecifiedUnits ( SVGLengthType unitType )
        {
            if ( unitType == mUnitType )
            {
                return; // No conversion needed
            }

            // Convert from user units to the specified unit type
            switch ( unitType )
            {
            case SVGLengthType::NUMBER:
            case SVGLengthType::PX:
                // User units are px, so no conversion needed
                mValueInSpecifiedUnits = mValue;
                break;

            case SVGLengthType::CM:
                // 1 inch = 2.54 cm, 1 inch = 96 px (at 96 DPI)
                // So 1 cm = 96/2.54 px
                mValueInSpecifiedUnits = mValue * 2.54f / 96.0f;
                break;

            case SVGLengthType::MM:
                // 1 inch = 25.4 mm, 1 inch = 96 px (at 96 DPI)
                // So 1 mm = 96/25.4 px
                mValueInSpecifiedUnits = mValue * 25.4f / 96.0f;
                break;

            case SVGLengthType::IN:
                // 1 inch = 96 px (at 96 DPI)
                mValueInSpecifiedUnits = mValue / 96.0f;
                break;

            case SVGLengthType::PT:
                // 1 inch = 72 pt, 1 inch = 96 px (at 96 DPI)
                // So 1 pt = 96/72 px = 4/3 px
                mValueInSpecifiedUnits = mValue * 72.0f / 96.0f;
                break;

            case SVGLengthType::PC:
                // 1 inch = 6 pc, 1 inch = 96 px (at 96 DPI)
                // So 1 pc = 96/6 px = 16 px
                mValueInSpecifiedUnits = mValue * 6.0f / 96.0f;
                break;

            case SVGLengthType::PERCENTAGE:
                // Percentage conversion requires viewport context
                // For now, assume 100% = 100 user units (this should be context-dependent)
                mValueInSpecifiedUnits = mValue;
                break;

            case SVGLengthType::EMS:
            case SVGLengthType::EXS:
            {
                // Font-relative units require font context
                // For now, assume 1em = 16px (default font size)
                float fontBase = ( unitType == SVGLengthType::EMS ) ? 16.0f : 8.0f; // ex is typically half of em
                mValueInSpecifiedUnits = mValue / fontBase;
            }
            break;

            default:
                // Unknown unit type, don't convert
                return;
            }

            // Update the unit type and value string representation
            mUnitType = unitType;

            // Update the string representation
            UpdateValueAsString();
        }

        void SVGLength::newValueSpecifiedUnits ( SVGLengthType unitType, float valueInSpecifiedUnits )
        {
            // Set the unit type and value in specified units
            mUnitType = unitType;
            mValueInSpecifiedUnits = valueInSpecifiedUnits;

            // Convert to user units (px) for the mValue field
            switch ( unitType )
            {
            case SVGLengthType::NUMBER:
            case SVGLengthType::PX:
                // User units are px, so no conversion needed
                mValue = valueInSpecifiedUnits;
                break;

            case SVGLengthType::CM:
                // 1 cm = 96/2.54 px (at 96 DPI)
                mValue = valueInSpecifiedUnits * 96.0f / 2.54f;
                break;

            case SVGLengthType::MM:
                // 1 mm = 96/25.4 px (at 96 DPI)
                mValue = valueInSpecifiedUnits * 96.0f / 25.4f;
                break;

            case SVGLengthType::IN:
                // 1 inch = 96 px (at 96 DPI)
                mValue = valueInSpecifiedUnits * 96.0f;
                break;

            case SVGLengthType::PT:
                // 1 pt = 96/72 px = 4/3 px
                mValue = valueInSpecifiedUnits * 96.0f / 72.0f;
                break;

            case SVGLengthType::PC:
                // 1 pc = 96/6 px = 16 px
                mValue = valueInSpecifiedUnits * 96.0f / 6.0f;
                break;

            case SVGLengthType::PERCENTAGE:
                // Percentage conversion requires viewport context
                // For now, assume 100% = 100 user units
                mValue = valueInSpecifiedUnits;
                break;

            case SVGLengthType::EMS:
            case SVGLengthType::EXS:
            {
                // Font-relative units require font context
                float fontBase = ( unitType == SVGLengthType::EMS ) ? 16.0f : 8.0f;
                mValue = valueInSpecifiedUnits * fontBase;
            }
            break;

            default:
                // Unknown unit type
                mValue = valueInSpecifiedUnits;
                break;
            }

            // Update the string representation
            UpdateValueAsString();
        }
    }
}
