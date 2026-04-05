/*
Copyright (C) 2019,2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_CANVAS_H
#define AEONGUI_CANVAS_H
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "aeongui/Platform.hpp"
#include "aeongui/DrawType.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/Attribute.hpp"
#include "aeongui/Matrix2x3.hpp"
#include "aeongui/TextLayout.hpp"
namespace AeonGUI
{
    class Path;
    /** @brief Abstract 2D rendering surface.
     *
     *  Provides a pure virtual interface for drawing paths, images, and text
     *  onto a pixel buffer.  Concrete implementations (e.g. CairoCanvas)
     *  supply the actual rendering back-end.
     */
    class Canvas
    {
    public:
        /** @brief Resize the rendering viewport.
         *  @param aWidth  New width in pixels.
         *  @param aHeight New height in pixels.
         */
        virtual void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) = 0;
        /** @brief Get a pointer to the raw pixel data.
         *  @return Pointer to BGRA pixel data, or nullptr if empty.
         */
        virtual const uint8_t* GetPixels() const = 0;
        /** @brief Get the width of the canvas in pixels.
         *  @return Width in pixels.
         */
        virtual size_t GetWidth() const = 0;
        /** @brief Get the height of the canvas in pixels.
         *  @return Height in pixels.
         */
        virtual size_t GetHeight() const = 0;
        /** @brief Get the stride (bytes per row) of the pixel buffer.
         *  @return Stride in bytes.
         */
        virtual size_t GetStride() const = 0;
        /** @brief Clear the canvas to transparent. */
        virtual void Clear() = 0;
        /** @brief Set the fill color.
         *  @param aColor The fill color to set.
         */
        virtual void SetFillColor ( const ColorAttr& aColor ) = 0;
        /** @brief Get the current fill color.
         *  @return Reference to the current fill color.
         */
        virtual const ColorAttr& GetFillColor() const = 0;
        /** @brief Set the stroke color.
         *  @param aColor The stroke color to set.
         */
        virtual void SetStrokeColor ( const ColorAttr& aColor ) = 0;
        /** @brief Get the current stroke color.
         *  @return Reference to the current stroke color.
         */
        virtual const ColorAttr& GetStrokeColor() const = 0;
        /** @brief Set the stroke width.
         *  @param aWidth The stroke width in user units.
         */
        virtual void SetStrokeWidth ( double aWidth ) = 0;
        /** @brief Get the current stroke width.
         *  @return Stroke width in user units.
         */
        virtual double GetStrokeWidth () const = 0;
        /** @brief Set the stroke opacity.
         *  @param aWidth Opacity value in the range [0.0, 1.0].
         */
        virtual void SetStrokeOpacity ( double aWidth ) = 0;
        /** @brief Get the current stroke opacity.
         *  @return Opacity in [0.0, 1.0].
         */
        virtual double GetStrokeOpacity () const = 0;
        /** @brief Set the fill opacity.
         *  @param aWidth Opacity value in the range [0.0, 1.0].
         */
        virtual void SetFillOpacity ( double aWidth ) = 0;
        /** @brief Get the current fill opacity.
         *  @return Opacity in [0.0, 1.0].
         */
        virtual double GetFillOpacity () const = 0;
        /** @brief Set the global opacity.
         *  @param aWidth Opacity value in the range [0.0, 1.0].
         */
        virtual void SetOpacity ( double aWidth ) = 0;
        /** @brief Get the current global opacity.
         *  @return Opacity in [0.0, 1.0].
         */
        virtual double GetOpacity () const = 0;
        /** @brief Draw a path using the current fill and stroke settings.
         *  @param aPath The path to draw.
         */
        virtual void Draw ( const Path& aPath ) = 0;
        /** @brief Draw a raster image.
         *  @param aPixels      Pointer to source BGRA pixel data.
         *  @param aImageWidth  Width of the source image in pixels.
         *  @param aImageHeight Height of the source image in pixels.
         *  @param aImageStride Stride of the source image in bytes.
         *  @param aX           Destination X coordinate.
         *  @param aY           Destination Y coordinate.
         *  @param aWidth       Destination width in user units.
         *  @param aHeight      Destination height in user units.
         *  @param aOpacity     Opacity for the image [0.0, 1.0].
         */
        virtual void DrawImage ( const uint8_t* aPixels,
                                 size_t aImageWidth,
                                 size_t aImageHeight,
                                 size_t aImageStride,
                                 double aX,
                                 double aY,
                                 double aWidth,
                                 double aHeight,
                                 double aOpacity ) = 0;
        /** Draw text at the given position using the specified font description and size.
         *  @param aText The UTF-8 text string to render.
         *  @param aX The x coordinate for the text start position.
         *  @param aY The y coordinate for the text baseline.
         *  @param aFontFamily Font family name (e.g. "sans-serif").
         *  @param aFontSize Font size in CSS pixels.
         *  @param aFontWeight CSS font weight (400 = normal, 700 = bold).
         *  @param aFontStyle 0 = normal, 1 = italic, 2 = oblique.
         */
        virtual void DrawText ( const std::string& aText, double aX, double aY,
                                const std::string& aFontFamily, double aFontSize,
                                int aFontWeight, int aFontStyle ) = 0;
        /** Measure the width of text with the given font parameters.
         *  @param aText       The UTF-8 text string to measure.
         *  @param aFontFamily Font family name.
         *  @param aFontSize   Font size in CSS pixels.
         *  @param aFontWeight CSS font weight.
         *  @param aFontStyle  Font style (0=normal, 1=italic, 2=oblique).
         *  @return The logical width in CSS pixels.
         */
        virtual double MeasureText ( const std::string& aText,
                                     const std::string& aFontFamily, double aFontSize,
                                     int aFontWeight, int aFontStyle ) const = 0;
        /** Draw text along a path.
         *  Each glyph is positioned and rotated to follow the path.
         *  @param aText        The UTF-8 text string to render.
         *  @param aPath        The path to follow.
         *  @param aStartOffset Starting distance along the path in user units.
         *  @param aFontFamily  Font family name.
         *  @param aFontSize    Font size in CSS pixels.
         *  @param aFontWeight  CSS font weight (400 = normal, 700 = bold).
         *  @param aFontStyle   0 = normal, 1 = italic, 2 = oblique.
         *  @param aReverse     If true, render glyph order against the path direction.
         *  @param aClosed      If true, treat the path as closed for offset wrapping.
         */
        virtual void DrawTextOnPath ( const std::string& aText,
                                      const Path& aPath,
                                      double aStartOffset,
                                      const std::string& aFontFamily, double aFontSize,
                                      int aFontWeight, int aFontStyle,
                                      bool aReverse = false, bool aClosed = false ) = 0;
        /** @brief Set the SVG viewBox and preserveAspectRatio.
         *  @param aViewBox The viewBox rectangle.
         *  @param aPreserveAspectRatio How to align and scale.
         */
        virtual void SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio ) = 0;
        /** @brief Replace the current transformation matrix.
         *  @param aMatrix The new 2x3 transformation matrix.
         */
        virtual void SetTransform ( const Matrix2x3& aMatrix ) = 0;
        /** @brief Pre-multiply the current transformation matrix.
         *  @param aMatrix The matrix to concatenate.
         */
        virtual void Transform ( const Matrix2x3& aMatrix ) = 0;
        /** @brief Save the current graphics state (transform, clipping, etc.).
         */
        virtual void Save() = 0;
        /** @brief Restore the previously saved graphics state.
         */
        virtual void Restore() = 0;
        /** @brief Get the native rendering surface handle.
         *  @return Pointer to the underlying surface (e.g. cairo_surface_t).
         */
        virtual void* GetNativeSurface() const = 0;
        /** @brief Begin an offscreen group for filter/compositing.
         *
         *  All subsequent drawing is redirected to a temporary surface.
         *  Call PopGroup() to retrieve and composite the result.
         */
        virtual void PushGroup() = 0;
        /** @brief End an offscreen group and composite back.
         *
         *  Pops the group pushed by PushGroup() and paints it back
         *  to the underlying surface.
         */
        virtual void PopGroup() = 0;
        /** @brief Apply a drop-shadow filter to the current group content.
         *
         *  Must be called between PushGroup() and PopGroup().
         *  The implementation captures the group, creates a blurred shadow,
         *  offsets it, and composites it under the original content.
         *
         *  @param aDx            Horizontal offset of the shadow.
         *  @param aDy            Vertical offset of the shadow.
         *  @param aStdDeviationX Horizontal Gaussian blur standard deviation.
         *  @param aStdDeviationY Vertical Gaussian blur standard deviation.
         *  @param aFloodColor    Shadow color.
         *  @param aFloodOpacity  Shadow opacity [0.0, 1.0].
         */
        virtual void ApplyDropShadow ( double aDx, double aDy,
                                       double aStdDeviationX, double aStdDeviationY,
                                       const Color& aFloodColor, double aFloodOpacity ) = 0;
        /** @brief Enable or disable hit-testing mode.
         *
         *  When true, filter effects (PushGroup/PopGroup/ApplyDropShadow)
         *  are skipped to avoid expensive pixel processing during
         *  elementFromPoint traversals.
         *  @param aHitTesting True to enable hit-testing mode, false to disable.
         */
        void SetHitTesting ( bool aHitTesting )
        {
            mHitTesting = aHitTesting;
        }
        /** @brief Query whether the canvas is in hit-testing mode.
         *  @return True if hit-testing is active.
         */
        bool IsHitTesting() const
        {
            return mHitTesting;
        }
        /** @brief Set the current pick ID for subsequent Draw calls.
         *
         *  When non-zero, Draw(path) also fills the path on the
         *  internal pick buffer using this value.  Set to 0 to
         *  disable pick rendering for non-geometry nodes.
         *  @param aPickId The pick ID to assign (0 disables pick rendering).
         */
        void SetPickId ( uint8_t aPickId )
        {
            mPickId = aPickId;
        }
        /** @brief Read the pick ID at the given viewport coordinates.
         *  @param aX X coordinate in viewport pixels.
         *  @param aY Y coordinate in viewport pixels.
         *  @return Pick ID at that pixel, or 0 if empty / out of bounds.
         */
        virtual uint8_t PickAtPoint ( double aX, double aY ) const = 0;
        /** @brief Clear the pick buffer and reset for a new frame. */
        virtual void ResetPick() = 0;
        /** @brief Set a device-space clip rectangle on both render and pick surfaces.
         *
         *  Must be called between Save() and Restore().
         *  @param aX X of clip rect (device pixels).
         *  @param aY Y of clip rect (device pixels).
         *  @param aWidth  Width (device pixels).
         *  @param aHeight Height (device pixels).
         */
        virtual void SetClipRect ( double aX, double aY, double aWidth, double aHeight ) = 0;
        /** @brief Create a new Path object suitable for this canvas backend.
         *  @return A new empty Path instance.
         */
        virtual std::unique_ptr<Path> CreatePath() const = 0;
        /** @brief Virtual destructor. */
        virtual ~Canvas() = 0;

        /** @brief Device-space bounding box for a pick-tracked element. */
        struct PickBounds
        {
            double x1{0}; ///< Left edge (device pixels).
            double y1{0}; ///< Top edge (device pixels).
            double x2{0}; ///< Right edge (device pixels).
            double y2{0}; ///< Bottom edge (device pixels).
        };
        /** @brief Get the cached device-space bounds for a pick ID.
         *  @param aId Pick ID (1-255).
         *  @return Reference to the cached bounds.
         */
        const PickBounds& GetPickBounds ( uint8_t aId ) const
        {
            return mPickBounds[aId];
        }
    protected:
        bool mHitTesting{false};                  ///< True when in hit-testing mode.
        uint8_t mPickId{0};                       ///< Current pick ID for Draw calls.
        std::array<PickBounds, 256> mPickBounds{}; ///< Cached device-space bounds per pick ID.
        double mViewportWidth{0};                 ///< Current SVG viewport width for percent resolution.
        double mViewportHeight{0};                ///< Current SVG viewport height for percent resolution.
        std::vector<std::pair<double, double>> mViewportStack; ///< Saved viewport dimensions.
    public:
        /** @brief Push a new SVG viewport for percentage resolution.
         *  @param aWidth  Viewport width in user units.
         *  @param aHeight Viewport height in user units.
         */
        void PushViewport ( double aWidth, double aHeight )
        {
            mViewportStack.emplace_back ( mViewportWidth, mViewportHeight );
            mViewportWidth  = aWidth;
            mViewportHeight = aHeight;
        }
        /** @brief Pop the SVG viewport, restoring the previous one. */
        void PopViewport()
        {
            if ( !mViewportStack.empty() )
            {
                mViewportWidth  = mViewportStack.back().first;
                mViewportHeight = mViewportStack.back().second;
                mViewportStack.pop_back();
            }
        }
        /** @brief Get the current SVG viewport width.
         *  @return Viewport width in user units.
         */
        double GetViewportWidth() const
        {
            return mViewportWidth;
        }
        /** @brief Get the current SVG viewport height.
         *  @return Viewport height in user units.
         */
        double GetViewportHeight() const
        {
            return mViewportHeight;
        }
    };
}
#endif
