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
#ifndef AEONGUI_HTMLIMAGEELEMENT_H
#define AEONGUI_HTMLIMAGEELEMENT_H

#include "HTMLElement.hpp"
#include "aeongui/RasterImage.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;img&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/embedded-content.html#htmlimageelement
         *
         *  Acts as a CSS replaced element: contributes its decoded
         *  intrinsic dimensions to the layout engine and paints itself
         *  into the content box at draw time.
         */
        class HTMLImageElement : public HTMLElement
        {
        public:
            HTMLImageElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLImageElement() final;

            AEONGUI_DLL void DrawStart ( Canvas& aCanvas ) const override;

            /// Lazily load the bitmap referenced by `src`, resolving
            /// relative paths against the owner document.  Returns
            /// true when the image is decoded and ready to paint.
            AEONGUI_DLL bool EnsureImageLoaded() const;

            /// Intrinsic width/height of the decoded image.  Returns
            /// 0 until the image is successfully loaded.
            AEONGUI_DLL uint32_t naturalWidth()  const;
            AEONGUI_DLL uint32_t naturalHeight() const;

        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;

        private:
            DOMString mSrc;
            mutable RasterImage mRasterImage;
            mutable bool mImageLoadAttempted{false};
        };
    }
}
#endif
