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
#include "aeongui/dom/Text.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/DrawType.hpp"
#include "aeongui/StyleSheet.hpp"
#include "PangoTextLayout.hpp"
#ifdef AEONGUI_USE_SKIA
#include "SkiaPath.hpp"
#else
#include "CairoPath.hpp"
#endif
#include <libcss/libcss.h>
#include <pango/pango.h>
#include <array>
#include <cctype>

namespace AeonGUI
{
    namespace DOM
    {
        HTMLElement::HTMLElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : Element { aTagName, std::move ( aAttributes ), aParent } {}
        HTMLElement::~HTMLElement() = default;

        namespace
        {
#ifdef AEONGUI_USE_SKIA
            using BackendPath = SkiaPath;
#else
            using BackendPath = CairoPath;
#endif

            /// Build a closed-rectangle path (M H V H Z) from two
            /// corners and emit it on the canvas using the current
            /// fill color.
            void FillRect ( Canvas& aCanvas, double aX0, double aY0,
                            double aX1, double aY1 )
            {
                if ( aX1 <= aX0 || aY1 <= aY0 )
                {
                    return;
                }
                std::array<DrawType, 10> commands{};
                size_t i = 0;
                commands[i++] = static_cast<uint64_t> ( 'M' );
                commands[i++] = aX0;
                commands[i++] = aY0;
                commands[i++] = static_cast<uint64_t> ( 'H' );
                commands[i++] = aX1;
                commands[i++] = static_cast<uint64_t> ( 'V' );
                commands[i++] = aY1;
                commands[i++] = static_cast<uint64_t> ( 'H' );
                commands[i++] = aX0;
                commands[i++] = static_cast<uint64_t> ( 'Z' );

                BackendPath path;
                path.Construct ( commands.data(), i );
                aCanvas.Draw ( path );
            }

            /// Map a CSS border-width result to layout pixels using
            /// the same THIN/MEDIUM/THICK convention as the layout
            /// engine, so paint and layout agree on edge thickness.
            float BorderEdgePx ( uint8_t aStyleResult, uint8_t aWidthResult,
                                 css_fixed aLength, css_unit aUnit )
            {
                if ( aStyleResult == CSS_BORDER_STYLE_NONE ||
                     aStyleResult == CSS_BORDER_STYLE_HIDDEN )
                {
                    return 0.0f;
                }
                switch ( aWidthResult )
                {
                case CSS_BORDER_WIDTH_THIN:
                    return 1.0f;
                case CSS_BORDER_WIDTH_MEDIUM:
                    return 3.0f;
                case CSS_BORDER_WIDTH_THICK:
                    return 5.0f;
                case CSS_BORDER_WIDTH_WIDTH:
                    if ( aUnit == CSS_UNIT_PX )
                    {
                        return static_cast<float> ( FIXTOFLT ( aLength ) );
                    }
                    return 0.0f;
                default:
                    return 0.0f;
                }
            }

            /// Resolve a border-color result to a paint color, falling
            /// back to the computed `color` for INHERIT / CURRENT_COLOR
            /// (CSS spec: border-color initial value is currentColor).
            /// Returns false when the resolved color is fully transparent
            /// and nothing should be painted.
            bool ResolveBorderColor ( const css_computed_style* aStyle,
                                      uint8_t aColorResult, css_color aColor,
                                      css_color& aOut )
            {
                if ( aColorResult == CSS_BORDER_COLOR_COLOR )
                {
                    aOut = aColor;
                }
                else
                {
                    css_color current{};
                    css_computed_color ( aStyle, &current );
                    aOut = current;
                }
                return aOut != 0;
            }
        }

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

            const ColorAttr previous_fill = aCanvas.GetFillColor();
            bool fill_dirty = false;

            const double x0 = mLayoutBox.x;
            const double y0 = mLayoutBox.y;
            const double x1 = mLayoutBox.x + mLayoutBox.width;
            const double y1 = mLayoutBox.y + mLayoutBox.height;

            // Background fill covers the entire border box.  Only
            // paint when the author set an explicit color; CURRENT_COLOR
            // and transparent fills are skipped for now.
            css_color css_bg{};
            uint8_t bg_type = css_computed_background_color ( style, &css_bg );
            if ( bg_type == CSS_BACKGROUND_COLOR_COLOR && css_bg != 0 )
            {
                aCanvas.SetFillColor ( ColorAttr{ Color{ static_cast<uint32_t> ( css_bg ) } } );
                fill_dirty = true;
                FillRect ( aCanvas, x0, y0, x1, y1 );
            }

            // Borders.  Each edge is read independently and painted as
            // an axis-aligned filled rectangle inside the border box.
            // Mitred corners are achieved by extending the top/bottom
            // edges across the full width while the left/right edges
            // only cover the inner vertical strip.  This keeps each
            // edge a single solid-color rectangle, matching libcss's
            // CSS_BORDER_STYLE_SOLID semantics; non-solid styles fall
            // back to solid in this slice.
            css_fixed length{};
            css_unit unit{};

            uint8_t top_w_r = css_computed_border_top_width    ( style, &length, &unit );
            float   top_px  = BorderEdgePx (
                                  css_computed_border_top_style ( style ),
                                  top_w_r, length, unit );
            uint8_t right_w_r = css_computed_border_right_width  ( style, &length, &unit );
            float   right_px  = BorderEdgePx (
                                    css_computed_border_right_style ( style ),
                                    right_w_r, length, unit );
            uint8_t bottom_w_r = css_computed_border_bottom_width ( style, &length, &unit );
            float   bottom_px  = BorderEdgePx (
                                     css_computed_border_bottom_style ( style ),
                                     bottom_w_r, length, unit );
            uint8_t left_w_r = css_computed_border_left_width   ( style, &length, &unit );
            float   left_px  = BorderEdgePx (
                                   css_computed_border_left_style ( style ),
                                   left_w_r, length, unit );

            auto paint_edge = [&] ( float aWidth,
                                    uint8_t ( *aColorGetter ) ( const css_computed_style*, css_color* ),
                                    double aEx0, double aEy0, double aEx1, double aEy1 )
            {
                if ( aWidth <= 0.0f )
                {
                    return;
                }
                css_color border_color{};
                uint8_t color_result = aColorGetter ( style, &border_color );
                css_color resolved{};
                if ( !ResolveBorderColor ( style, color_result, border_color, resolved ) )
                {
                    return;
                }
                aCanvas.SetFillColor ( ColorAttr{ Color{ static_cast<uint32_t> ( resolved ) } } );
                fill_dirty = true;
                FillRect ( aCanvas, aEx0, aEy0, aEx1, aEy1 );
            };

            paint_edge ( top_px,    css_computed_border_top_color,
                         x0, y0, x1, y0 + top_px );
            paint_edge ( bottom_px, css_computed_border_bottom_color,
                         x0, y1 - bottom_px, x1, y1 );
            paint_edge ( left_px,   css_computed_border_left_color,
                         x0, y0 + top_px, x0 + left_px, y1 - bottom_px );
            paint_edge ( right_px,  css_computed_border_right_color,
                         x1 - right_px, y0 + top_px, x1, y1 - bottom_px );

            // Inline text content.  We concatenate every Text child into
            // a single run, lay it out through PangoTextLayout with the
            // content-box width as the wrap constraint (matches what
            // HTMLLayoutEngine's measure callback used), and then walk
            // the resulting lines so each one is painted via the
            // backend's DrawText at its own baseline.  This keeps the
            // canvas API minimal — both Cairo and Skia already build
            // their own per-call PangoLayout for DrawText, so paint
            // and measure stay agreement-by-construction.
            css_select_results* mutable_results = GetComputedStyles();
            css_computed_style* mutable_style = mutable_results
                                                ? mutable_results->styles[CSS_PSEUDO_ELEMENT_NONE]
                                                : nullptr;
            if ( mutable_style )
            {
                std::string concatenated;
                for ( const auto& child : childNodes() )
                {
                    if ( child->nodeType() != Node::TEXT_NODE )
                    {
                        continue;
                    }
                    concatenated += static_cast<const Text*> ( child.get() )->wholeText();
                }
                bool only_ws = true;
                for ( char c : concatenated )
                {
                    if ( !std::isspace ( static_cast<unsigned char> ( c ) ) )
                    {
                        only_ws = false;
                        break;
                    }
                }
                if ( !concatenated.empty() && !only_ws )
                {
                    const std::string family  = GetCSSFontFamily ( mutable_style );
                    const double      size    = GetCSSFontSize   ( mutable_style );
                    const int         weight  = GetCSSFontWeight ( mutable_style );
                    const int         style_n = GetCSSFontStyle  ( mutable_style );

                    css_color text_color{};
                    css_computed_color ( style, &text_color );
                    if ( text_color != 0 )
                    {
                        aCanvas.SetFillColor (
                            ColorAttr{ Color{ static_cast<uint32_t> ( text_color ) } } );
                        fill_dirty = true;

                        PangoTextLayout layout;
                        layout.SetFontFamily ( family );
                        layout.SetFontSize   ( size );
                        layout.SetFontWeight ( weight );
                        layout.SetFontStyle  ( style_n );
                        if ( mLayoutBox.contentWidth > 0.0f )
                        {
                            layout.SetWrapWidth ( mLayoutBox.contentWidth );
                        }
                        layout.SetText ( concatenated );

                        PangoLayout* pango_layout = layout.GetPangoLayout();
                        PangoLayoutIter* iter = pango_layout_get_iter ( pango_layout );
                        do
                        {
                            PangoLayoutLine* line =
                                pango_layout_iter_get_line_readonly ( iter );
                            const int baseline_pango =
                                pango_layout_iter_get_baseline ( iter );
                            if ( !line || line->length <= 0 )
                            {
                                continue;
                            }
                            const std::string line_text = concatenated.substr (
                                                              static_cast<size_t> ( line->start_index ),
                                                              static_cast<size_t> ( line->length ) );
                            // Skip whitespace-only lines (the trailing
                            // newline of a wrapped paragraph would
                            // otherwise paint an empty box outline).
                            bool line_only_ws = true;
                            for ( char c : line_text )
                            {
                                if ( !std::isspace ( static_cast<unsigned char> ( c ) ) )
                                {
                                    line_only_ws = false;
                                    break;
                                }
                            }
                            if ( line_only_ws )
                            {
                                continue;
                            }
                            const double baseline_px =
                                mLayoutBox.contentY +
                                static_cast<double> ( baseline_pango ) / PANGO_SCALE;
                            aCanvas.DrawText ( line_text,
                                               mLayoutBox.contentX,
                                               baseline_px,
                                               family, size, weight, style_n );
                        }
                        while ( pango_layout_iter_next_line ( iter ) );
                        pango_layout_iter_free ( iter );
                    }
                }
            }

            if ( fill_dirty )
            {
                aCanvas.SetFillColor ( previous_fill );
            }
        }
    }
}
