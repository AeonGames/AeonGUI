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
#ifndef AEONGUI_SVGIMAGEELEMENT_H
#define AEONGUI_SVGIMAGEELEMENT_H

#include "SVGGraphicsElement.hpp"
#include "SVGAnimatedLength.hpp"
#include "SVGAnimatedString.hpp"
#include "SVGAnimatedPreserveAspectRatio.hpp"
#include "aeongui/RasterImage.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class SVGImageElement : public SVGGraphicsElement
        {
        public:
            SVGImageElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGImageElement() final;

            const SVGAnimatedLength& x() const;
            const SVGAnimatedLength& y() const;
            const SVGAnimatedLength& width() const;
            const SVGAnimatedLength& height() const;
            const SVGAnimatedString& href() const;
            const SVGAnimatedPreserveAspectRatio& preserveAspectRatio() const;
            const DOMString& crossOrigin() const;
            const DOMString& decoding() const;

            void DrawStart ( Canvas& aCanvas ) const final;

        private:
            void ParseAttributes ( const AttributeMap& aAttributes );
            bool EnsureImageLoaded() const;

            SVGAnimatedLength mX{};
            SVGAnimatedLength mY{};
            SVGAnimatedLength mWidth{};
            SVGAnimatedLength mHeight{};
            SVGAnimatedString mHref{};
            SVGAnimatedPreserveAspectRatio mPreserveAspectRatio{};
            DOMString mCrossOrigin{};
            DOMString mDecoding{"auto"};
            mutable RasterImage mRasterImage{};
            mutable bool mImageLoadAttempted{false};
        };
    }
}

#endif
