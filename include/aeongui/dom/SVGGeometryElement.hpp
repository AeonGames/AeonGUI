/*
Copyright (C) 2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "SVGGraphicsElement.hpp"
// Path type should be selectable and should match Canvas type
#include "aeongui/CairoPath.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Base class for SVG shape elements that describe geometry.
         *  @see https://www.w3.org/TR/SVG2/types.html#InterfaceSVGGeometryElement
         */
        class SVGGeometryElement : public SVGGraphicsElement
        {
        public:
            /** @brief Construct an SVGGeometryElement.
             *  @param aTagName    Tag name.
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGGeometryElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGGeometryElement() override;
            /** @brief Draw the geometry path onto the canvas.
             *  @param aCanvas Target canvas.
             */
            void DrawStart ( Canvas& aCanvas ) const final;
            /** @brief Get the path data for this geometry element.
             *  @return Const reference to the CairoPath.
             */
            const CairoPath& GetPath() const
            {
                return mPath;
            }
        protected:
            /** @brief Rebuild the cached path with animated attribute values.
             *
             *  Derived classes override this to reconstruct the path when
             *  child path-modifying animations (e.g. rx/ry on rect) are active.
             */
            virtual void RebuildAnimatedPath() const {}
            mutable CairoPath mPath; ///< Cached Cairo path for this geometry.
        };
    }
}
#endif
