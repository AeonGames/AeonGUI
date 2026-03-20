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
#ifndef AEONGUI_SVGANIMATEDENUMERATION_HPP
#define AEONGUI_SVGANIMATEDENUMERATION_HPP

#include <cstdint>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated enumeration attribute (SVG).
         *
         *  Wraps a base and animated integer value for SVG enumeration attributes.
         */
        class DLL SVGAnimatedEnumeration
        {
        public:
            /** @brief Default constructor. */
            SVGAnimatedEnumeration();
            /** @brief Destructor. */
            ~SVGAnimatedEnumeration();
            /** @brief Get the base value. */
            int32_t baseVal() const;
            /** @brief Get the animated value. */
            int32_t animVal() const;
        private:
            int32_t mBaseVal;
            int32_t mAnimVal;
        };
    }
}

#endif
