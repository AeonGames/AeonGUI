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
#include "aeongui/dom/SVGAnimatedString.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGAnimatedString::SVGAnimatedString() = default;
        SVGAnimatedString::~SVGAnimatedString() = default;

        DOMString& SVGAnimatedString::baseVal()
        {
            return mBaseVal;
        }

        const DOMString& SVGAnimatedString::baseVal() const
        {
            return mBaseVal;
        }

        DOMString& SVGAnimatedString::animVal()
        {
            return mAnimVal;
        }

        const DOMString& SVGAnimatedString::animVal() const
        {
            return mAnimVal;
        }
    }
}
