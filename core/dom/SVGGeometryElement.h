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
#ifndef AEONGUI_SVGGEOMETRYELEMENT_H
#define AEONGUI_SVGGEOMETRYELEMENT_H

#include <vector>
#include "SVGGraphicsElement.h"
// Path type should be selectable and should match Canvas type
#include "aeongui/CairoPath.h"

namespace AeonGUI
{
    namespace DOM
    {
        class SVGGeometryElement : public SVGGraphicsElement
        {
        public:
            SVGGeometryElement ( const std::string& aTagName, const AttributeMap& aAttributes );
            ~SVGGeometryElement() override;
            void DrawStart ( Canvas& aCanvas ) const final;
        protected:
            CairoPath mPath;
        };
    }
}
#endif
