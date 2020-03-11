/*
Copyright (C) 2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGSTOPELEMENT_H
#define AEONGUI_SVGSTOPELEMENT_H

#include "SVGElement.h"

namespace AeonGUI
{
    namespace DOM
    {

        class SVGStopElement : public SVGElement
        {
        public:
            SVGStopElement ( const std::string& aTagName, const AttributeMap& aAttributes );
            ~SVGStopElement() final;
        };
    }
}
#endif
