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
#ifndef AEONGUI_SVGANIMATEDPRESERVEASPECTRATIO_HPP
#define AEONGUI_SVGANIMATEDPRESERVEASPECTRATIO_HPP

#include "aeongui/Attribute.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Animated preserveAspectRatio attribute (SVG).
         *
         *  Wraps base and animated PreserveAspectRatio values.
         */
        class DLL SVGAnimatedPreserveAspectRatio
        {
        public:
            /** @brief Default constructor. */
            SVGAnimatedPreserveAspectRatio();
            /** @brief Destructor. */
            ~SVGAnimatedPreserveAspectRatio();

            /** @brief Get the base value (mutable).
             *  @return Reference to the base PreserveAspectRatio. */
            PreserveAspectRatio& baseVal();
            /** @brief Get the base value (const).
             *  @return Const reference to the base PreserveAspectRatio. */
            const PreserveAspectRatio& baseVal() const;
            /** @brief Get the animated value (mutable).
             *  @return Reference to the animated PreserveAspectRatio. */
            PreserveAspectRatio& animVal();
            /** @brief Get the animated value (const).
             *  @return Const reference to the animated PreserveAspectRatio. */
            const PreserveAspectRatio& animVal() const;

        private:
            PreserveAspectRatio mBaseVal{};
            PreserveAspectRatio mAnimVal{};
        };
    }
}

#endif
