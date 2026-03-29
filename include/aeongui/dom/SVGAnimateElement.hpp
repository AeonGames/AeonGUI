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
#ifndef AEONGUI_SVGANIMATEELEMENT_H
#define AEONGUI_SVGANIMATEELEMENT_H

#include "SVGAnimationElement.hpp"
#include "aeongui/Color.hpp"
#include <vector>

namespace AeonGUI
{
    namespace DOM
    {
        class SVGAnimateElement : public SVGAnimationElement
        {
        public:
            SVGAnimateElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGAnimateElement() override;
            void ApplyToCanvas ( Canvas& aCanvas ) const override;
            bool IsGeometryAnimation() const;
            bool IsPathAnimation() const;
            double GetInterpolatedValue() const;
        private:
            Color InterpolateColor ( double aProgress ) const;
            double InterpolateNumber ( double aProgress ) const;
            void ApplyGeometryToCanvas ( Canvas& aCanvas ) const;
            bool mIsColorAnimation{false};
            bool mIsGeometryAnimation{false};
            bool mIsPathAnimation{false};
            double mOriginalValue{0.0};
            double mAnchorX{0.0};
            double mAnchorY{0.0};
            std::vector<Color> mColorValues;
            std::vector<double> mNumericValues;
        };
    }
}
#endif
