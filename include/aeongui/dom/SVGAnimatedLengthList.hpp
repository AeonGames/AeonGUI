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
#ifndef AEONGUI_SVGANIMATEDLENGTHLIST_HPP
#define AEONGUI_SVGANIMATEDLENGTHLIST_HPP

#include "SVGLengthList.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated length list attribute (SVG).
         *
         *  Wraps base and animated SVGLengthList values.
         */
        class DLL SVGAnimatedLengthList
        {
        public:
            /** @brief Default constructor. */
            SVGAnimatedLengthList();
            /** @brief Destructor. */
            ~SVGAnimatedLengthList();

            /** @brief Get the base value (mutable). */
            SVGLengthList& baseVal();
            /** @brief Get the base value (const). */
            const SVGLengthList& baseVal() const;
            /** @brief Get the animated value (const). */
            const SVGLengthList& animVal() const;

        private:
            SVGLengthList mBaseVal;
            SVGLengthList mAnimVal;
        };
    }
}

#endif
