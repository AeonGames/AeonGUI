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

#include "aeongui/Cursor.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "aeongui/Canvas.hpp"
#include "aeongui/RasterImage.hpp"
#include "aeongui/dom/Document.hpp"

namespace AeonGUI
{
    Cursor::Cursor() = default;
    Cursor::~Cursor() = default;

    void Cursor::Clear()
    {
        mPixels.clear();
        mPixels.shrink_to_fit();
        mBacking.clear();
        mBacking.shrink_to_fit();
        mWidth = 0;
        mHeight = 0;
        mHotspotX = 0;
        mHotspotY = 0;
        mSavedX = mSavedY = mSavedW = mSavedH = 0;
        mHasBacking = false;
    }

    void Cursor::SetEnabled ( bool aEnabled )
    {
        if ( mEnabled == aEnabled )
        {
            return;
        }
        mEnabled = aEnabled;
        // Drop any pending backing store: it is no longer meaningful and would
        // otherwise be written back unexpectedly the next time the cursor is
        // re-enabled.
        mHasBacking = false;
    }

    void Cursor::DiscardBackingStore() noexcept
    {
        mHasBacking = false;
    }

    void Cursor::SetSource ( const RasterImage& aImage,
                             int32_t aHotspotX, int32_t aHotspotY )
    {
        if ( !aImage.IsLoaded() ||
             aImage.GetPixelFormat() != RasterImage::PixelFormat::RGBA8 )
        {
            throw std::runtime_error ( "Cursor::SetSource: RasterImage is not a loaded RGBA8 image." );
        }
        const uint32_t w = aImage.GetWidth();
        const uint32_t h = aImage.GetHeight();
        const size_t srcStride = aImage.GetStride();
        const uint8_t* src = aImage.GetPixels();
        if ( w == 0 || h == 0 || src == nullptr )
        {
            throw std::runtime_error ( "Cursor::SetSource: empty RasterImage." );
        }

        mWidth = w;
        mHeight = h;
        mHotspotX = aHotspotX;
        mHotspotY = aHotspotY;
        mPixels.assign ( static_cast<size_t> ( w ) * h * 4u, 0 );
        // Convert RGBA8 (straight alpha) -> BGRA8 (premultiplied).
        for ( uint32_t y = 0; y < h; ++y )
        {
            const uint8_t* srcRow = src + y * srcStride;
            uint8_t* dstRow = mPixels.data() + static_cast<size_t> ( y ) * w * 4u;
            for ( uint32_t x = 0; x < w; ++x )
            {
                const uint8_t r = srcRow[x * 4u + 0u];
                const uint8_t g = srcRow[x * 4u + 1u];
                const uint8_t b = srcRow[x * 4u + 2u];
                const uint8_t a = srcRow[x * 4u + 3u];
                dstRow[x * 4u + 0u] = static_cast<uint8_t> ( ( static_cast<uint32_t> ( b ) * a + 127u ) / 255u );
                dstRow[x * 4u + 1u] = static_cast<uint8_t> ( ( static_cast<uint32_t> ( g ) * a + 127u ) / 255u );
                dstRow[x * 4u + 2u] = static_cast<uint8_t> ( ( static_cast<uint32_t> ( r ) * a + 127u ) / 255u );
                dstRow[x * 4u + 3u] = a;
            }
        }
        // Drop any stale backing store; its rect no longer matches.
        mBacking.clear();
        mBacking.shrink_to_fit();
        mHasBacking = false;
    }

    void Cursor::SetSource ( const std::string& aSvgUrl,
                             uint32_t aWidth, uint32_t aHeight )
    {
        if ( aWidth == 0 || aHeight == 0 )
        {
            throw std::runtime_error ( "Cursor::SetSource: SVG render size must be > 0." );
        }

        // Render the SVG into a transient canvas, then snapshot its BGRA pixels.
        std::unique_ptr<Canvas> canvas = Canvas::Create ( aWidth, aHeight );
        canvas->Clear();
        DOM::Document doc;
        doc.Load ( aSvgUrl );
        doc.Draw ( *canvas );

        const uint8_t* pixels = canvas->GetPixels();
        const size_t srcStride = canvas->GetStride();
        if ( pixels == nullptr )
        {
            throw std::runtime_error ( "Cursor::SetSource: SVG render produced no pixels." );
        }

        mWidth = aWidth;
        mHeight = aHeight;
        // SVG hotspot is fixed at the SVG origin (0, 0) by design.
        mHotspotX = 0;
        mHotspotY = 0;
        mPixels.assign ( static_cast<size_t> ( aWidth ) * aHeight * 4u, 0 );
        // Canvas pixels are already premultiplied BGRA — copy row by row in
        // case the canvas stride differs from a tightly packed buffer.
        const size_t dstStride = static_cast<size_t> ( aWidth ) * 4u;
        for ( uint32_t y = 0; y < aHeight; ++y )
        {
            std::memcpy ( mPixels.data() + y * dstStride,
                          pixels + y * srcStride,
                          dstStride );
        }
        mBacking.clear();
        mBacking.shrink_to_fit();
        mHasBacking = false;
    }

    void Cursor::Composite ( uint8_t* aBuffer,
                             size_t aBufferWidth,
                             size_t aBufferHeight,
                             size_t aBufferStride,
                             int32_t aMouseX,
                             int32_t aMouseY )
    {
        if ( !mEnabled || !HasSource() || aBuffer == nullptr )
        {
            return;
        }

        // Top-left of the cursor image in buffer coordinates.
        const int32_t cx = aMouseX - mHotspotX;
        const int32_t cy = aMouseY - mHotspotY;

        // Clip the cursor rect against the buffer.
        const int32_t bw = static_cast<int32_t> ( aBufferWidth );
        const int32_t bh = static_cast<int32_t> ( aBufferHeight );
        const int32_t x0 = std::max ( cx, 0 );
        const int32_t y0 = std::max ( cy, 0 );
        const int32_t x1 = std::min ( cx + static_cast<int32_t> ( mWidth ),  bw );
        const int32_t y1 = std::min ( cy + static_cast<int32_t> ( mHeight ), bh );
        if ( x0 >= x1 || y0 >= y1 )
        {
            // Fully off-screen: no backing store, nothing to blit.
            mHasBacking = false;
            return;
        }

        const int32_t w = x1 - x0;
        const int32_t h = y1 - y0;
        // Offset into the cursor image where the visible region starts.
        const int32_t sx = x0 - cx;
        const int32_t sy = y0 - cy;

        // Snapshot the destination pixels before we overwrite them.
        const size_t backingStride = static_cast<size_t> ( w ) * 4u;
        mBacking.assign ( backingStride * h, 0 );
        for ( int32_t row = 0; row < h; ++row )
        {
            const uint8_t* dstRow = aBuffer + ( static_cast<size_t> ( y0 + row ) ) * aBufferStride
                                    + static_cast<size_t> ( x0 ) * 4u;
            std::memcpy ( mBacking.data() + static_cast<size_t> ( row ) * backingStride,
                          dstRow, backingStride );
        }
        mSavedX = x0;
        mSavedY = y0;
        mSavedW = w;
        mSavedH = h;
        mHasBacking = true;

        // Alpha-blend cursor (premultiplied src-over) into the buffer.
        const size_t cursorStride = static_cast<size_t> ( mWidth ) * 4u;
        for ( int32_t row = 0; row < h; ++row )
        {
            const uint8_t* srcRow = mPixels.data()
                                    + static_cast<size_t> ( sy + row ) * cursorStride
                                    + static_cast<size_t> ( sx ) * 4u;
            uint8_t* dstRow = aBuffer + ( static_cast<size_t> ( y0 + row ) ) * aBufferStride
                              + static_cast<size_t> ( x0 ) * 4u;
            for ( int32_t col = 0; col < w; ++col )
            {
                const uint32_t sb = srcRow[col * 4u + 0u];
                const uint32_t sg = srcRow[col * 4u + 1u];
                const uint32_t sr = srcRow[col * 4u + 2u];
                const uint32_t sa = srcRow[col * 4u + 3u];
                if ( sa == 0u )
                {
                    continue;
                }
                if ( sa == 255u )
                {
                    dstRow[col * 4u + 0u] = static_cast<uint8_t> ( sb );
                    dstRow[col * 4u + 1u] = static_cast<uint8_t> ( sg );
                    dstRow[col * 4u + 2u] = static_cast<uint8_t> ( sr );
                    dstRow[col * 4u + 3u] = 255u;
                    continue;
                }
                const uint32_t inv = 255u - sa;
                const uint32_t db = dstRow[col * 4u + 0u];
                const uint32_t dg = dstRow[col * 4u + 1u];
                const uint32_t dr = dstRow[col * 4u + 2u];
                const uint32_t da = dstRow[col * 4u + 3u];
                dstRow[col * 4u + 0u] = static_cast<uint8_t> ( sb + ( db * inv + 127u ) / 255u );
                dstRow[col * 4u + 1u] = static_cast<uint8_t> ( sg + ( dg * inv + 127u ) / 255u );
                dstRow[col * 4u + 2u] = static_cast<uint8_t> ( sr + ( dr * inv + 127u ) / 255u );
                dstRow[col * 4u + 3u] = static_cast<uint8_t> ( sa + ( da * inv + 127u ) / 255u );
            }
        }
    }

    void Cursor::Restore ( uint8_t* aBuffer,
                           size_t aBufferWidth,
                           size_t aBufferHeight,
                           size_t aBufferStride )
    {
        if ( !mEnabled || !HasSource() || !mHasBacking || aBuffer == nullptr )
        {
            return;
        }
        // Defensive: if the buffer shrank since the snapshot, clamp the
        // restore region (do not write out of bounds).
        const int32_t bw = static_cast<int32_t> ( aBufferWidth );
        const int32_t bh = static_cast<int32_t> ( aBufferHeight );
        const int32_t x0 = std::max ( mSavedX, 0 );
        const int32_t y0 = std::max ( mSavedY, 0 );
        const int32_t x1 = std::min ( mSavedX + mSavedW, bw );
        const int32_t y1 = std::min ( mSavedY + mSavedH, bh );
        if ( x0 < x1 && y0 < y1 )
        {
            const size_t backingStride = static_cast<size_t> ( mSavedW ) * 4u;
            const int32_t srcOffsetX = x0 - mSavedX;
            const int32_t srcOffsetY = y0 - mSavedY;
            const size_t copyBytes = static_cast<size_t> ( x1 - x0 ) * 4u;
            for ( int32_t row = 0; row < ( y1 - y0 ); ++row )
            {
                std::memcpy ( aBuffer + ( static_cast<size_t> ( y0 + row ) ) * aBufferStride
                              + static_cast<size_t> ( x0 ) * 4u,
                              mBacking.data() + static_cast<size_t> ( srcOffsetY + row ) * backingStride
                              + static_cast<size_t> ( srcOffsetX ) * 4u,
                              copyBytes );
            }
        }
        mHasBacking = false;
    }
}
