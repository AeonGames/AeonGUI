/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGANIMATEDSTRING_HPP
#define AEONGUI_SVGANIMATEDSTRING_HPP

#include "DOMString.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated string attribute (SVG).
         *
         *  Wraps base and animated DOMString values.
         */
        class SVGAnimatedString
        {
        public:
            /** @brief Default constructor. */
            DLL SVGAnimatedString();
            /** @brief Destructor. */
            DLL ~SVGAnimatedString();

            /** @brief Get the base value (mutable). */
            DLL DOMString& baseVal();
            /** @brief Get the base value (const). */
            DLL const DOMString& baseVal() const;
            /** @brief Get the animated value (mutable). */
            DLL DOMString& animVal();
            /** @brief Get the animated value (const). */
            DLL const DOMString& animVal() const;

        private:
            DOMString mBaseVal{};
            DOMString mAnimVal{};
        };
    }
}

#endif
