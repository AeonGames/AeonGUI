/*
Copyright (C) 2019,2020,2023-2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGSVGELEMENT_H
#define AEONGUI_SVGSVGELEMENT_H

#include "SVGGraphicsElement.hpp"
#include "aeongui/Attribute.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class SVGSVGElement : public SVGGraphicsElement
        {
        public:
            SVGSVGElement ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent );
            ~SVGSVGElement() final;
            void DrawStart ( Canvas& aCanvas ) const final;
        private:
            // Attributes
            double mWidth{};
            double mHeight{};
            ViewBox mViewBox{};
            PreserveAspectRatio mPreserveAspectRatio{};
        };
    }
}
#endif
