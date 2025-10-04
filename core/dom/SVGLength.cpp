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

namespace AeonGUI
{
    namespace DOM
    {
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

        const DOMString& SVGLength::valueAsString() const
        {
            return mValueAsString;
        }

        void SVGLength::valueAsString ( const DOMString& value )
        {
            mValue = std::stof ( value );
            mValueAsString = value;
        }

        float SVGLength::valueInSpecifiedUnits() const
        {
            return mValueInSpecifiedUnits;
        }

        void SVGLength::convertToSpecifiedUnits ( SVGLengthType /*unitType*/ )
        {
        }

        void SVGLength::newValueSpecifiedUnits ( SVGLengthType /*unitType*/, float /*valueInSpecifiedUnits*/ )
        {
        }
    }
}
