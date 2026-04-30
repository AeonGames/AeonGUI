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
#ifndef AEONGUI_CURSOR_H
#define AEONGUI_CURSOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    class RasterImage;

    /** @brief Software-composited custom cursor.
     *
     *  A Cursor caches its image as a premultiplied BGRA pixel buffer
     *  (built either from a @ref RasterImage or by rasterizing an SVG
     *  document once at @ref SetSource time) and composites itself
     *  directly into a target BGRA buffer using a classic save/restore
     *  backing store: the pixels under the cursor are snapshotted before
     *  the cursor is blitted, then restored before the next blit so that
     *  cursor motion never dirties the document.
     *
     *  Disabled by default. When disabled, @ref Composite and @ref Restore
     *  are single-branch no-ops, so an unused Cursor adds no rendering cost.
     *
     *  @note Hotspot semantics:
     *  - Raster cursors take an explicit @c hotspotX, @c hotspotY pair (the
     *    pixel that aligns with the system cursor position).
     *  - SVG cursors always use the SVG origin (0, 0) as hotspot — author
     *    your SVG accordingly.
     */
    class Cursor
    {
    public:
        AEONGUI_DLL Cursor();
        AEONGUI_DLL ~Cursor();

        Cursor ( const Cursor& ) = delete;
        Cursor& operator= ( const Cursor& ) = delete;
        Cursor ( Cursor&& ) = delete;
        Cursor& operator= ( Cursor&& ) = delete;

        /** @brief Replace the cursor image with a copy of @p aImage.
         *  @param aImage    Decoded raster image (any format @ref RasterImage supports).
         *  @param aHotspotX X-pixel of the cursor "tip" (relative to image).
         *  @param aHotspotY Y-pixel of the cursor "tip" (relative to image).
         *  @throws std::runtime_error if the image is not loaded or has an
         *          unsupported pixel format.
         */
        AEONGUI_DLL void SetSource ( const RasterImage& aImage,
                                     int32_t aHotspotX, int32_t aHotspotY );

        /** @brief Replace the cursor image by loading and rasterizing an SVG.
         *
         *  The SVG is loaded through the standard @ref DOM::Document
         *  pipeline (so the application's @ref ResourceLoader is consulted)
         *  and rendered once into the cursor cache at the requested size.
         *  The hotspot is fixed at the SVG origin (0, 0).
         *  @param aSvgUrl   URL or filesystem path to the SVG document.
         *  @param aWidth    Render width in pixels (must be > 0).
         *  @param aHeight   Render height in pixels (must be > 0).
         *  @throws std::runtime_error if loading or rendering fails.
         */
        AEONGUI_DLL void SetSource ( const std::string& aSvgUrl,
                                     uint32_t aWidth, uint32_t aHeight );

        /** @brief Drop the current cursor image and release its caches. */
        AEONGUI_DLL void Clear();

        /** @brief Enable or disable cursor compositing.
         *  Disabled cursors are not blitted and do not maintain a backing store.
         */
        AEONGUI_DLL void SetEnabled ( bool aEnabled );

        /** @return true if the cursor is currently enabled. */
        [[nodiscard]] AEONGUI_DLL bool IsEnabled() const noexcept
        {
            return mEnabled;
        }

        /** @return true if a cursor image has been set. */
        [[nodiscard]] AEONGUI_DLL bool HasSource() const noexcept
        {
            return mWidth != 0 && mHeight != 0;
        }

        /** @return Cursor image width in pixels. */
        [[nodiscard]] AEONGUI_DLL uint32_t GetWidth() const noexcept
        {
            return mWidth;
        }

        /** @return Cursor image height in pixels. */
        [[nodiscard]] AEONGUI_DLL uint32_t GetHeight() const noexcept
        {
            return mHeight;
        }

        /** @return Hotspot X pixel relative to the cursor image. */
        [[nodiscard]] AEONGUI_DLL int32_t GetHotspotX() const noexcept
        {
            return mHotspotX;
        }

        /** @return Hotspot Y pixel relative to the cursor image. */
        [[nodiscard]] AEONGUI_DLL int32_t GetHotspotY() const noexcept
        {
            return mHotspotY;
        }

        /** @brief Composite the cursor onto a BGRA pixel buffer at @p (aMouseX, aMouseY).
         *
         *  Snapshots the destination pixels currently under the cursor into
         *  the internal backing store (so they can be restored later) and
         *  alpha-blends the cursor over the buffer. No-op if the cursor is
         *  disabled or has no source.
         *
         *  @param aBuffer Pointer to a BGRA8 (premultiplied) pixel buffer.
         *  @param aBufferWidth  Buffer width in pixels.
         *  @param aBufferHeight Buffer height in pixels.
         *  @param aBufferStride Buffer row stride in bytes.
         *  @param aMouseX Mouse X position (in buffer pixels) — the hotspot
         *                 will be placed here.
         *  @param aMouseY Mouse Y position (in buffer pixels).
         */
        AEONGUI_DLL void Composite ( uint8_t* aBuffer,
                                     size_t aBufferWidth,
                                     size_t aBufferHeight,
                                     size_t aBufferStride,
                                     int32_t aMouseX,
                                     int32_t aMouseY );

        /** @brief Restore the pixels previously snapshotted by @ref Composite.
         *
         *  No-op if the cursor is disabled, has no source, or has no
         *  outstanding backing store. The buffer dimensions/stride must
         *  match those passed to the matching @ref Composite call.
         */
        AEONGUI_DLL void Restore ( uint8_t* aBuffer,
                                   size_t aBufferWidth,
                                   size_t aBufferHeight,
                                   size_t aBufferStride );

        /** @brief Drop any outstanding backing store without writing it back.
         *
         *  Call this after the underlying buffer has been fully repainted
         *  from another source (e.g. after a full canvas redraw), so that
         *  the next @ref Composite snapshots the freshly drawn pixels
         *  rather than restoring stale ones.
         */
        AEONGUI_DLL void DiscardBackingStore() noexcept;

        /** @return true if a backing store snapshot is currently held. */
        [[nodiscard]] AEONGUI_DLL bool HasBackingStore() const noexcept
        {
            return mHasBacking;
        }

    private:
        /// Cursor image, premultiplied BGRA8, tightly packed (stride = mWidth*4).
        std::vector<uint8_t> mPixels;
        /// Snapshot of the destination pixels currently behind the cursor,
        /// premultiplied BGRA8, tightly packed (stride = mSavedW*4).
        std::vector<uint8_t> mBacking;
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        int32_t mHotspotX{0};
        int32_t mHotspotY{0};
        // Saved blit rectangle for the current backing-store snapshot.
        int32_t mSavedX{0};
        int32_t mSavedY{0};
        int32_t mSavedW{0};
        int32_t mSavedH{0};
        bool mEnabled{false};
        bool mHasBacking{false};
    };
}

#endif
