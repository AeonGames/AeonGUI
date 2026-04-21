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
#ifndef AEONGUI_HTMLLAYOUTENGINE_H
#define AEONGUI_HTMLLAYOUTENGINE_H

#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class HTMLElement;
    }

    /** @brief Computes layout for an HTML element subtree using Yoga.
     *
     *  The engine builds a parallel Yoga tree mirroring the HTMLElement
     *  subtree, copies a subset of libcss computed style properties
     *  (width, height, margin, padding, flex-direction, display) onto
     *  the Yoga nodes, runs YGNodeCalculateLayout, and writes the
     *  resulting border-box rectangles back onto each HTMLElement via
     *  HTMLElement::SetLayoutBox.
     *
     *  No painting is performed.  Coordinates are in CSS pixels with
     *  the root at (0, 0).
     */
    class HTMLLayoutEngine
    {
    public:
        /** @brief Construct an engine with a fresh Yoga config. */
        AEONGUI_DLL HTMLLayoutEngine();
        /** @brief Destructor. */
        AEONGUI_DLL ~HTMLLayoutEngine();

        HTMLLayoutEngine ( const HTMLLayoutEngine& ) = delete;
        HTMLLayoutEngine& operator= ( const HTMLLayoutEngine& ) = delete;

        /** @brief Compute layout for the given HTML subtree.
         *  @param aRoot          Root HTML element to lay out.  Non-HTML
         *                        descendants (e.g. inline SVG) are
         *                        treated as opaque leaf boxes whose
         *                        size comes from CSS only.
         *  @param aWidth         Available width in CSS pixels.
         *  @param aHeight        Available height in CSS pixels.
         */
        AEONGUI_DLL void Layout ( DOM::HTMLElement* aRoot, float aWidth, float aHeight );

    private:
        struct Impl;
        Impl* mImpl;
    };
}
#endif
