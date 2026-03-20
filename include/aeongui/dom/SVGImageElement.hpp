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
        /** @brief SVG image element for embedding raster images.
         *  @see https://www.w3.org/TR/SVG2/embedded.html#InterfaceSVGImageElement
         */
        class SVGImageElement : public SVGGraphicsElement
        {
        public:
            /** @brief Construct an SVGImageElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            DLL SVGImageElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            DLL ~SVGImageElement() final;

            /** @brief Get the animated x position. */
            DLL const SVGAnimatedLength& x() const;
            /** @brief Get the animated y position. */
            DLL const SVGAnimatedLength& y() const;
            /** @brief Get the animated width. */
            DLL const SVGAnimatedLength& width() const;
            /** @brief Get the animated height. */
            DLL const SVGAnimatedLength& height() const;
            /** @brief Get the animated href (image source URL). */
            DLL const SVGAnimatedString& href() const;
            /** @brief Get the animated preserveAspectRatio. */
            DLL const SVGAnimatedPreserveAspectRatio& preserveAspectRatio() const;
            /** @brief Get the crossOrigin attribute value. */
            DLL const DOMString& crossOrigin() const;
            /** @brief Get the decoding hint ("auto", "sync", or "async"). */
            DLL const DOMString& decoding() const;

            /** @brief Draw the embedded image onto the canvas.
             *  @param aCanvas Target canvas.
             */
            DLL void DrawStart ( Canvas& aCanvas ) const final;

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
