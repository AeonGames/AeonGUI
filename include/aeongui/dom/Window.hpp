/*
Copyright (C) 2019,2020,2023,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_WINDOW_H
#define AEONGUI_WINDOW_H
#include <cstdint>
#include <string>
#include "aeongui/Platform.hpp"
#include "aeongui/CairoCanvas.hpp"
#include "aeongui/dom/EventTarget.hpp"
#include "aeongui/dom/USVString.hpp"
#include "aeongui/dom/Location.hpp"
#include "aeongui/dom/Document.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Document;
        /** @brief Represents a display window in the AeonGUI framework.
         *
         *  Owns a Document, a CairoCanvas, and a Location.
         *  Renders the document into a pixel buffer that can be blitted
         *  to the screen.
         *  @see https://html.spec.whatwg.org/multipage/nav-history-apis.html#window
         */
        class Window : public EventTarget
        {
        public:
            /** @brief Default constructor. Creates an empty window. */
            DLL Window ();
            /** @brief Construct a window with the given viewport size.
             *  @param aWidth  Initial width in pixels.
             *  @param aHeight Initial height in pixels.
             */
            DLL Window ( uint32_t aWidth, uint32_t aHeight );
            /** @brief Destructor. */
            DLL ~Window () override final;
            /** @brief Resize the rendering viewport.
             *  @param aWidth  New width in pixels.
             *  @param aHeight New height in pixels.
             */
            DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight );
            /** @brief Get a pointer to the rendered pixel data.
             *  @return Pointer to BGRA pixel data.
             */
            DLL const uint8_t* GetPixels() const;
            /** @brief Get the window width in pixels.
             *  @return The width. */
            DLL size_t GetWidth() const;
            /** @brief Get the window height in pixels.
             *  @return The height. */
            DLL size_t GetHeight() const;
            /** @brief Get the stride (bytes per row) of the pixel buffer.
             *  @return The stride in bytes. */
            DLL size_t GetStride() const;
            /** @brief Render the current document to the internal canvas. */
            DLL void Draw();
            /**DOM Properties and Methods @{*/
            /** @brief Get the associated Document.
             *  @return Pointer to the Document, or nullptr.
             */
            DLL const Document* document() const;
            /** @brief Get the Location object.
             *  @return Reference to the window's Location.
             */
            DLL Location& location() const;
            /**@}*/
        private:
            void OnLocationChanged ( const Location& location );
            Location mLocation{std::bind ( &Window::OnLocationChanged, this, std::placeholders::_1 ) };
            Document mDocument{};
            CairoCanvas mCanvas{};
        };
    }
}
#endif
