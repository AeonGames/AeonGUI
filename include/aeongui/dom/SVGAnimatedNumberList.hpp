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
#ifndef AEONGUI_SVGANIMATEDNUMBERLIST_HPP
#define AEONGUI_SVGANIMATEDNUMBERLIST_HPP

#include "SVGNumberList.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated number list attribute (SVG).
         *
         *  Wraps base and animated SVGNumberList values.
         */
        class DLL SVGAnimatedNumberList
        {
        public:
            /** @brief Default constructor. */
            SVGAnimatedNumberList();
            /** @brief Destructor. */
            ~SVGAnimatedNumberList();

            /** @brief Get the base value (mutable). */
            SVGNumberList& baseVal();
            /** @brief Get the base value (const). */
            const SVGNumberList& baseVal() const;
            /** @brief Get the animated value (const). */
            const SVGNumberList& animVal() const;

        private:
            SVGNumberList mBaseVal;
            SVGNumberList mAnimVal;
        };
    }
}

#endif