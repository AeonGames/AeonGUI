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
#ifndef AEONGUI_SVGLENGTH_HPP
#define AEONGUI_SVGLENGTH_HPP

#include "DOMString.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        enum class SVGLengthType
        {
            UNKNOWN = 0,
            NUMBER = 1,
            PERCENTAGE = 2,
            EMS = 3,
            EXS = 4,
            PX = 5,
            CM = 6,
            MM = 7,
            IN = 8,
            PT = 9,
            PC = 10
        };

        class DLL SVGLength
        {
        public:
            SVGLength();
            ~SVGLength();
            /// @brief The type of the length. One of the SVGLengthType constants.
            SVGLengthType unitType() const;

            /// @brief The value as a floating point value, in user units.
            float value() const;
            float value ( float value );

            /// @brief The value as a string value, in the units expressed by unitType.
            const DOMString& valueAsString() const;
            const DOMString& valueAsString ( const DOMString& value );

            /// @brief The value as a floating point value, in the units expressed by unitType.
            float valueInSpecifiedUnits() const;
            float valueInSpecifiedUnits ( float value );

            /// @brief Preserve the same underlying stored value, but reset the stored unit identifier to the given unitType.
            void convertToSpecifiedUnits ( SVGLengthType unitType );

            /// @brief Reset the value as a number with an associated unitType, thereby replacing the values for all of the attributes on the object.
            void newValueSpecifiedUnits ( SVGLengthType unitType, float valueInSpecifiedUnits );
        private:
            void UpdateValueAsString();
            SVGLengthType mUnitType{SVGLengthType::UNKNOWN};
            float mValue{};
            float mValueInSpecifiedUnits{};
            DOMString mValueAsString{};
        };
    }
}
#endif
