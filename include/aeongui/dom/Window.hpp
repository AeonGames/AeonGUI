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
#include <array>
#include <functional>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "aeongui/Platform.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/dom/EventTarget.hpp"
#include "aeongui/dom/USVString.hpp"
#include "aeongui/dom/DOMString.hpp"
#include "aeongui/dom/Location.hpp"
#include "aeongui/dom/Document.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Document;
        /** @brief Represents a display window in the AeonGUI framework.
         *
         *  Owns a Document, a Canvas, and a Location.
         *  Renders the document into a pixel buffer that can be blitted
         *  to the screen.
         *  @see https://html.spec.whatwg.org/multipage/nav-history-apis.html#window
         */
        class Window : public EventTarget
        {
        public:
            /** @brief Default constructor. Creates an empty window. */
            AEONGUI_DLL Window ();
            /** @brief Construct a window with the given viewport size.
             *  @param aWidth  Initial width in pixels.
             *  @param aHeight Initial height in pixels.
             */
            AEONGUI_DLL Window ( uint32_t aWidth, uint32_t aHeight );
            /** @brief Destructor. */
            AEONGUI_DLL ~Window () override final;
            /** @brief Resize the rendering viewport.
             *  @param aWidth  New width in pixels.
             *  @param aHeight New height in pixels.
             */
            AEONGUI_DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight );
            /** @brief Get a pointer to the rendered pixel data.
             *  @return Pointer to BGRA pixel data.
             */
            AEONGUI_DLL const uint8_t* GetPixels() const;
            /** @brief Get the window width in pixels.
             *  @return The width. */
            AEONGUI_DLL size_t GetWidth() const;
            /** @brief Get the window height in pixels.
             *  @return The height. */
            AEONGUI_DLL size_t GetHeight() const;
            /** @brief Get the stride (bytes per row) of the pixel buffer.
             *  @return The stride in bytes. */
            AEONGUI_DLL size_t GetStride() const;
            /** @brief Render the current document to the internal canvas.
             *  @return true if a redraw was performed, false if the scene was clean.
             */
            AEONGUI_DLL bool Draw();
            /** @brief Advance animation time and update all animations.
             *  @param aDeltaTime Time elapsed since last frame, in seconds.
             */
            AEONGUI_DLL void Update ( double aDeltaTime );
            /**DOM Properties and Methods @{*/
            /** @brief Get the associated Document.
             *  @return Pointer to the Document, or nullptr.
             */
            AEONGUI_DLL const Document* document() const;
            /** @brief Get the Location object.
             *  @return Reference to the window's Location.
             */
            AEONGUI_DLL Location& location() const;
            /** @brief Callback invoked once per animation frame.
             *  Receives the current frame timestamp in milliseconds.
             *  @see https://html.spec.whatwg.org/multipage/imagebitmap-and-animations.html#animationframeprovider
             */
            using FrameRequestCallback = std::function<void ( double ) >;
            /** @brief Schedule @p aCallback to be invoked before the next frame.
             *
             *  Mirrors the HTML spec @c requestAnimationFrame method on Window.
             *  Callbacks scheduled while the current frame's callbacks are running
             *  are deferred to the next frame.
             *  @param aCallback Function to invoke. Receives the frame timestamp
             *                   in milliseconds.
             *  @return A handle that can be passed to cancelAnimationFrame to
             *          cancel the request. Always non-zero.
             */
            AEONGUI_DLL uint32_t requestAnimationFrame ( FrameRequestCallback aCallback );
            /** @brief Cancel a previously scheduled animation frame callback.
             *  @param aHandle Handle returned by requestAnimationFrame. Unknown
             *                 or already-fired handles are silently ignored.
             */
            AEONGUI_DLL void cancelAnimationFrame ( uint32_t aHandle );
            /**@}*/
            /**Input Handling @{*/
            /** @brief Handle a mouse move event from the platform.
             *  @param aX       X coordinate in viewport pixels.
             *  @param aY       Y coordinate in viewport pixels.
             *  @param aButtons Bitmask of currently pressed buttons.
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleMouseMove ( double aX, double aY, unsigned short aButtons = 0,
                                               bool aCtrlKey = false, bool aShiftKey = false,
                                               bool aAltKey = false, bool aMetaKey = false );
            /** @brief Handle a mouse button press event from the platform.
             *  @param aX       X coordinate in viewport pixels.
             *  @param aY       Y coordinate in viewport pixels.
             *  @param aButton  Button number (0=primary, 1=middle, 2=secondary).
             *  @param aButtons Bitmask of currently pressed buttons.
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleMouseDown ( double aX, double aY, short aButton = 0,
                                               unsigned short aButtons = 0,
                                               bool aCtrlKey = false, bool aShiftKey = false,
                                               bool aAltKey = false, bool aMetaKey = false );
            /** @brief Handle a mouse button release event from the platform.
             *  @param aX       X coordinate in viewport pixels.
             *  @param aY       Y coordinate in viewport pixels.
             *  @param aButton  Button number (0=primary, 1=middle, 2=secondary).
             *  @param aButtons Bitmask of currently pressed buttons.
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleMouseUp ( double aX, double aY, short aButton = 0,
                                             unsigned short aButtons = 0,
                                             bool aCtrlKey = false, bool aShiftKey = false,
                                             bool aAltKey = false, bool aMetaKey = false );
            /** @brief Handle a keyboard key down event from the platform.
             *  @param aKey      The key value string.
             *  @param aCode     The physical key code string.
             *  @param aLocation Key location (DOM_KEY_LOCATION_*).
             *  @param aRepeat   True if this is a key repeat.
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleKeyDown ( const DOMString& aKey, const DOMString& aCode,
                                             uint32_t aLocation = 0, bool aRepeat = false,
                                             bool aCtrlKey = false, bool aShiftKey = false,
                                             bool aAltKey = false, bool aMetaKey = false );
            /** @brief Handle a keyboard key up event from the platform.
             *  @param aKey      The key value string.
             *  @param aCode     The physical key code string.
             *  @param aLocation Key location (DOM_KEY_LOCATION_*).
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleKeyUp ( const DOMString& aKey, const DOMString& aCode,
                                           uint32_t aLocation = 0,
                                           bool aCtrlKey = false, bool aShiftKey = false,
                                           bool aAltKey = false, bool aMetaKey = false );
            /** @brief Handle a wheel/scroll event from the platform.
             *  @param aX       X coordinate in viewport pixels.
             *  @param aY       Y coordinate in viewport pixels.
             *  @param aDeltaX  Horizontal scroll amount.
             *  @param aDeltaY  Vertical scroll amount.
             *  @param aDeltaMode Unit indicator (DOM_DELTA_PIXEL, etc.).
             *  @param aButtons Bitmask of currently pressed buttons.
             *  @param aCtrlKey  Control key held.
             *  @param aShiftKey Shift key held.
             *  @param aAltKey   Alt key held.
             *  @param aMetaKey  Meta key held.
             */
            AEONGUI_DLL void HandleWheel ( double aX, double aY,
                                           double aDeltaX, double aDeltaY,
                                           uint32_t aDeltaMode = 0,
                                           unsigned short aButtons = 0,
                                           bool aCtrlKey = false, bool aShiftKey = false,
                                           bool aAltKey = false, bool aMetaKey = false );
            /**@}*/
        private:
            void OnLocationChanged ( const Location& location );
            Element* elementFromPoint ( double aX, double aY ) const;
            void FullDraw();
            void PartialDraw();
            void AssignPickIds();
            void CacheBounds();
            Element* mFocusedElement{nullptr}; ///< The currently focused element.
            Element* mHoverElement{nullptr};   ///< The element currently under the pointer.
            Element* mActiveElement{nullptr};  ///< The element currently being clicked (mousedown).
            Location mLocation{std::bind ( &Window::OnLocationChanged, this, std::placeholders::_1 ) };
            Document mDocument{};
            std::unique_ptr<Canvas> mCanvas;
            std::array<Element*, 256> mPickElements{}; ///< Pick ID → Element* map (0 = none).
            uint8_t mPickIdCounter{0}; ///< Number of pick IDs assigned this frame.
            /// Cached device-space bounds per element for dirty rect computation.
            std::unordered_map<const Element*, Canvas::PickBounds> mCachedBounds{};
            /// Pending animation frame callbacks (handle, callback). Drained each Update().
            std::vector<std::pair<uint32_t, FrameRequestCallback>> mAnimationFrameCallbacks{};
            uint32_t mNextAnimationFrameHandle{0};
            double mAnimationFrameTimestamp{0.0}; ///< Accumulated time in milliseconds.
        };
    }
}
#endif
