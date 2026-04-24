/*
Copyright (C) 2023-2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_STYLESHEET_H
#define AEONGUI_STYLESHEET_H

#include <memory>
#include <string>

extern "C"
{
    struct css_stylesheet;
    struct css_select_ctx;
    struct css_select_results;
    struct css_computed_style;
}

namespace AeonGUI
{
    class Canvas;
    namespace DOM
    {
        class Element;
    }
}

namespace AeonGUI
{
    /** @brief Custom deleter for css_stylesheet. */
    struct css_stylesheet_deleter
    {
        /** @brief Release a css_stylesheet.
         *  @param p Pointer to the stylesheet.
         */
        void operator() ( css_stylesheet* p );
    };

    /** @brief Custom deleter for css_select_ctx. */
    struct css_select_ctx_deleter
    {
        /** @brief Release a css_select_ctx.
         *  @param p Pointer to the selection context.
         */
        void operator() ( css_select_ctx* p );
    };

    /** @brief Custom deleter for css_select_results. */
    struct css_select_results_deleter
    {
        /** @brief Release css_select_results.
         *  @param p Pointer to the selection results.
         */
        void operator() ( css_select_results* p );
    };

    /** @brief Custom deleter for css_computed_style. */
    struct css_computed_style_deleter
    {
        /** @brief Release a css_computed_style.
         *  @param p Pointer to the computed style.
         */
        void operator() ( css_computed_style* p );
    };

    /** @brief Owning pointer to a libcss stylesheet. */
    using StyleSheetPtr = std::unique_ptr<css_stylesheet, css_stylesheet_deleter>;
    /** @brief Owning pointer to a libcss selection context. */
    using SelectCtxPtr = std::unique_ptr<css_select_ctx, css_select_ctx_deleter>;
    /** @brief Owning pointer to libcss selection results. */
    using SelectResultsPtr = std::unique_ptr<css_select_results, css_select_results_deleter>;
    /** @brief Owning pointer to a libcss computed style. */
    using ComputedStylePtr = std::unique_ptr<css_computed_style, css_computed_style_deleter>;

    /** @brief Extract the CSS font-family from a computed style.
     *  @param aStyle The computed CSS style.
     *  @return The font family name.
     */
    std::string GetCSSFontFamily ( css_computed_style* aStyle );
    /** @brief Extract the CSS font-size from a computed style.
     *  @param aStyle The computed CSS style.
     *  @return The font size in pixels.
     */
    double GetCSSFontSize ( css_computed_style* aStyle );
    /** @brief Extract the CSS font-weight from a computed style.
     *  @param aStyle The computed CSS style.
     *  @return The font weight (100–900).
     */
    int GetCSSFontWeight ( css_computed_style* aStyle );
    /** @brief Extract the CSS font-style from a computed style.
     *  @param aStyle The computed CSS style.
     *  @return 0 = normal, 1 = italic, 2 = oblique.
     */
    int GetCSSFontStyle ( css_computed_style* aStyle );
    /** @brief Logical text alignment, mirroring the CSS text-align
     *  property values relevant to the HTML painter.  Justify is
     *  intentionally folded into Left for now: Pango can render
     *  PANGO_ALIGN_LEFT cheaply and we don't yet do the line-end
     *  spacing adjustment justify requires.
     */
    enum class TextAlign
    {
        Left,
        Right,
        Center
    };
    /** @brief Extract the CSS text-align value from a computed style.
     *  @param aStyle The computed CSS style.
     *  @return Resolved alignment; LTR-default and unknown values map
     *          to TextAlign::Left.
     */
    TextAlign GetCSSTextAlign ( css_computed_style* aStyle );
    /** @brief Apply fill, stroke, and opacity CSS properties to a canvas.
     *  @param aCanvas The canvas to configure.
     *  @param aElement The element to resolve paint server URIs against.
     *  @param aStyle The computed CSS style.
     */
    void ApplyCSSPaintProperties ( Canvas& aCanvas, const DOM::Element& aElement, css_computed_style* aStyle );
}
#endif
