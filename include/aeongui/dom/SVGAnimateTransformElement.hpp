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
#ifndef AEONGUI_SVGANIMATETRANSFORMELEMENT_H
#define AEONGUI_SVGANIMATETRANSFORMELEMENT_H

#include "SVGAnimationElement.hpp"
#include "aeongui/Matrix2x3.hpp"
#include <vector>

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief SMIL \<animateTransform\> element.
         *
         *  Animates rotate, scale, translate, skewX, or skewY transforms
         *  with keyframe interpolation.
         *  @see https://www.w3.org/TR/SVG11/animate.html#AnimateTransformElement
         */
        class SVGAnimateTransformElement : public SVGAnimationElement
        {
        public:
            /** @brief The kind of transform being animated. */
            enum class TransformType { Translate, Scale, Rotate, SkewX, SkewY };

            /** @brief Construct an SVGAnimateTransformElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGAnimateTransformElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGAnimateTransformElement() override;
            /** @brief Apply the animated transform to the canvas.
             *  @param aCanvas The target canvas.
             */
            void ApplyToCanvas ( Canvas& aCanvas ) const override;
        private:
            static std::vector<double> ParseNumbers ( const std::string& aStr );
            Matrix2x3 ComputeTransform ( double aProgress ) const;
            TransformType mTransformType{TransformType::Rotate};
            bool mAdditive{false};
            std::vector<std::vector<double>> mKeyframes;
        };
    }
}
#endif
