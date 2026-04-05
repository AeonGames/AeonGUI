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
#include <iostream>
#include "aeongui/dom/SVGGeometryElement.hpp"
#ifdef AEONGUI_USE_SKIA
#include "SkiaPath.hpp"
#else
#include "CairoPath.hpp"
#endif
#include "aeongui/StyleSheet.hpp"
#include <libcss/libcss.h>
namespace AeonGUI
{
    namespace DOM
    {
        SVGGeometryElement::SVGGeometryElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent ) : SVGGraphicsElement ( aTagName, std::move ( aAttributes ), aParent ), mPath{std::make_unique <
#ifdef AEONGUI_USE_SKIA
                    SkiaPath
#else
                    CairoPath
#endif
                    > () }
        {
        }
        SVGGeometryElement::~SVGGeometryElement() = default;
        void SVGGeometryElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                ApplyCSSPaintProperties ( aCanvas, *this, results->styles[CSS_PSEUDO_ELEMENT_NONE] );
            }
            ApplyChildPaintAnimations ( aCanvas );
            ResolveViewportPercentages ( aCanvas );
            RebuildAnimatedPath();
            aCanvas.Draw ( *mPath );
        }
    }
}
