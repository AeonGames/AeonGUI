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
#ifndef AEONGUI_SVGANIMATEDLENGTH_HPP
#define AEONGUI_SVGANIMATEDLENGTH_HPP

#include "SVGLength.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated length attribute (SVG).
         *
         *  Wraps base and animated SVGLength values for SVG length attributes.
         */
        class AEONGUI_DLL SVGAnimatedLength
        {
        public:
            /** @brief Default constructor. */
            SVGAnimatedLength();
            /** @brief Destructor. */
            ~SVGAnimatedLength();

            /** @brief Get the base value (mutable).
             *  @return Reference to the base SVGLength. */
            SVGLength& baseVal();
            /** @brief Get the base value (const).
             *  @return Const reference to the base SVGLength. */
            const SVGLength& baseVal() const;
            /** @brief Get the animated value (mutable).
             *  @return Reference to the animated SVGLength. */
            SVGLength& animVal();
            /** @brief Get the animated value (const).
             *  @return Const reference to the animated SVGLength. */
            const SVGLength& animVal() const;

        private:
            SVGLength mBaseVal;
            SVGLength mAnimVal;
        };
    }
}

#endif
