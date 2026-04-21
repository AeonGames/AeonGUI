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
#include "aeongui/dom/HTMLElement.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/DrawType.hpp"
#include "aeongui/StyleSheet.hpp"
#ifdef AEONGUI_USE_SKIA
#include "SkiaPath.hpp"
#else
#include "CairoPath.hpp"
#endif
#include <libcss/libcss.h>
#include <array>

namespace AeonGUI
{
    namespace DOM
    {
        HTMLElement::HTMLElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : Element { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLElement::~HTMLElement() = default;

        void HTMLElement::DrawStart ( Canvas& aCanvas ) const
        {
            // Border-box sized 0 contributes nothing to render.  This
            // also short-circuits display: none, which the layout
            // engine collapses to a 0x0 box.
            if ( mLayoutBox.width <= 0.0f || mLayoutBox.height <= 0.0f )
            {
                return;
            }

            css_select_results* results = GetComputedStyles();
            if ( !results || !results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                return;
            }
            const css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];

            css_color css_bg{};
            uint8_t bg_type = css_computed_background_color ( style, &css_bg );
            // Only paint when the author set an explicit color; we don't
            // synthesize a transparent default fill.  CURRENT_COLOR
            // could be supported later by reading css_computed_color.
            if ( bg_type != CSS_BACKGROUND_COLOR_COLOR || css_bg == 0 )
            {
                return;
            }

            // libcss gives back the same 0xAARRGGBB packing AeonGUI's
            // Color uses, so we can construct directly.
            const ColorAttr previous_fill = aCanvas.GetFillColor();
            aCanvas.SetFillColor ( ColorAttr{ Color{ static_cast<uint32_t> ( css_bg ) } } );

            // Build a closed-rectangle path in document coordinates
            // matching the laid-out border box.
            std::array<DrawType, 10> commands{};
            const double x0 = mLayoutBox.x;
            const double y0 = mLayoutBox.y;
            const double x1 = mLayoutBox.x + mLayoutBox.width;
            const double y1 = mLayoutBox.y + mLayoutBox.height;
            size_t i = 0;
            commands[i++] = static_cast<uint64_t> ( 'M' );
            commands[i++] = x0;
            commands[i++] = y0;
            commands[i++] = static_cast<uint64_t> ( 'H' );
            commands[i++] = x1;
            commands[i++] = static_cast<uint64_t> ( 'V' );
            commands[i++] = y1;
            commands[i++] = static_cast<uint64_t> ( 'H' );
            commands[i++] = x0;
            commands[i++] = static_cast<uint64_t> ( 'Z' );

#ifdef AEONGUI_USE_SKIA
            SkiaPath path;
#else
            CairoPath path;
#endif
            path.Construct ( commands.data(), i );
            aCanvas.Draw ( path );

            aCanvas.SetFillColor ( previous_fill );
        }
    }
}
