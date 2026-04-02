/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGLENGTH_HPP
#define AEONGUI_SVGLENGTH_HPP

#include "DOMString.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SVG length unit type constants. */
        enum class SVGLengthType
        {
            UNKNOWN = 0,    ///< Unknown or unsupported unit.
            NUMBER = 1,     ///< Unitless number.
            PERCENTAGE = 2, ///< Percentage.
            EMS = 3,        ///< Font-relative em units.
            EXS = 4,        ///< x-height units.
            PX = 5,         ///< CSS pixels.
            CM = 6,         ///< Centimeters.
            MM = 7,         ///< Millimeters.
            IN = 8,         ///< Inches.
            PT = 9,         ///< Points (1/72 of an inch).
            PC = 10         ///< Picas (1/6 of an inch).
        };

        /** @brief Represents an SVG length value with a unit.
         *
         *  Stores a numeric value and its associated unit type.
         *  @see https://www.w3.org/TR/SVG2/types.html#InterfaceSVGLength
         */
        class DLL SVGLength
        {
        public:
            /** @brief Default constructor. */
            SVGLength();
            /** @brief Destructor. */
            ~SVGLength();
            /// @brief The type of the length. One of the SVGLengthType constants.
            /// @return The unit type.
            SVGLengthType unitType() const;

            /// @brief The value as a floating point value, in user units.
            /// @return The value in user units.
            float value() const;
            /// @brief Set the value in user units.
            /// @param value New value.
            /// @return The new value.
            float value ( float value );

            /// @brief The value as a string value, in the units expressed by unitType.
            /// @return The string representation.
            const DOMString& valueAsString() const;
            /// @brief Set the value from a string.
            /// @param value New string value.
            /// @return The new string value.
            const DOMString& valueAsString ( const DOMString& value );

            /// @brief The value as a floating point value, in the units expressed by unitType.
            /// @return The value in specified units.
            float valueInSpecifiedUnits() const;
            /// @brief Set the value in specified units.
            /// @param value New value.
            /// @return The new value.
            float valueInSpecifiedUnits ( float value );

            /// @brief Preserve the same underlying stored value, but reset the stored unit identifier to the given unitType.
            /// @param unitType The target unit type.
            void convertToSpecifiedUnits ( SVGLengthType unitType );

            /// @brief Reset the value as a number with an associated unitType, thereby replacing the values for all of the attributes on the object.
            /// @param unitType The new unit type.
            /// @param valueInSpecifiedUnits The new value in the specified units.
            void newValueSpecifiedUnits ( SVGLengthType unitType, float valueInSpecifiedUnits );
        private:
            void UpdateValueAsString();
            SVGLengthType mUnitType{SVGLengthType::UNKNOWN};
            float mValue{};
            float mValueInSpecifiedUnits{};
            PRIVATE_TEMPLATE_MEMBERS_START
            DOMString mValueAsString{};
            PRIVATE_TEMPLATE_MEMBERS_END
        };
    }
}
#endif
