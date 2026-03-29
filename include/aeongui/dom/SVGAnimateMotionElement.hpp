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
#ifndef AEONGUI_SVGANIMATEMOTIONELEMENT_H
#define AEONGUI_SVGANIMATEMOTIONELEMENT_H

#include "SVGAnimationElement.hpp"
#include <vector>

namespace AeonGUI
{
    namespace DOM
    {
        class SVGAnimateMotionElement : public SVGAnimationElement
        {
        public:
            SVGAnimateMotionElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGAnimateMotionElement() override;
            void ApplyToCanvas ( Canvas& aCanvas ) const override;
        private:
            struct PathPoint
            {
                double x;
                double y;
                double cumulativeLength;
            };
            void LinearizePath ( const std::string& aPathData );
            std::vector<PathPoint> mPathPoints;
            double mTotalLength{0.0};
        };
    }
}
#endif
