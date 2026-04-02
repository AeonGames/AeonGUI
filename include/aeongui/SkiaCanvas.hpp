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
#ifndef AEONGUI_SKIACANVAS_H
#define AEONGUI_SKIACANVAS_H
#include <cstdint>
#include <string>
#include <vector>
#include <include/core/SkRefCnt.h>
#include "aeongui/Canvas.hpp"

class SkSurface;
class SkCanvas;

namespace AeonGUI
{
    /** @brief Skia-based Canvas implementation.
     *
     *  Renders 2D geometry, text, and images into a CPU-side pixel buffer
     *  using Google Skia.  Selected at build time with @c -DAEONGUI_BACKEND=Skia.
     */
    class SkiaCanvas : public Canvas
    {
    public:
        /** @brief Construct an empty (zero-size) SkiaCanvas. */
        DLL SkiaCanvas ();
        /** @brief Construct a SkiaCanvas with the given viewport dimensions.
         *  @param aWidth  Viewport width in pixels.
         *  @param aHeight Viewport height in pixels.
         */
        DLL SkiaCanvas ( uint32_t aWidth, uint32_t aHeight );
        DLL ~SkiaCanvas() final;
        DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) final;
        DLL const uint8_t* GetPixels() const final;
        DLL size_t GetWidth() const final;
        DLL size_t GetHeight() const final;
        DLL size_t GetStride() const final;
        DLL void Clear() final;
        DLL void Draw ( const Path& aPath ) final;
        DLL void DrawImage ( const uint8_t* aPixels,
                             size_t aImageWidth,
                             size_t aImageHeight,
                             size_t aImageStride,
                             double aX,
                             double aY,
                             double aWidth,
                             double aHeight,
                             double aOpacity ) final;
        DLL void DrawText ( const std::string& aText, double aX, double aY,
                            const std::string& aFontFamily, double aFontSize,
                            int aFontWeight, int aFontStyle ) final;
        DLL double MeasureText ( const std::string& aText,
                                 const std::string& aFontFamily, double aFontSize,
                                 int aFontWeight, int aFontStyle ) const final;
        DLL void DrawTextOnPath ( const std::string& aText,
                                  const Path& aPath,
                                  double aStartOffset,
                                  const std::string& aFontFamily, double aFontSize,
                                  int aFontWeight, int aFontStyle,
                                  bool aReverse = false, bool aClosed = false ) final;
        DLL void SetFillColor ( const ColorAttr& aColor ) final;
        DLL const ColorAttr& GetFillColor() const final;
        DLL void SetStrokeColor ( const ColorAttr& aColor ) final;
        DLL const ColorAttr& GetStrokeColor() const final;
        DLL void SetStrokeWidth ( double aWidth ) final;
        DLL double GetStrokeWidth () const final;
        DLL void SetStrokeOpacity ( double aWidth ) final;
        DLL double GetStrokeOpacity () const final;
        DLL void SetFillOpacity ( double aWidth ) final;
        DLL double GetFillOpacity () const final;
        DLL void SetOpacity ( double aWidth ) final;
        DLL double GetOpacity () const final;
        DLL void SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio ) final;
        DLL void SetTransform ( const Matrix2x3& aMatrix ) final;
        DLL void Transform ( const Matrix2x3& aMatrix ) final;
        DLL void Save() final;
        DLL void Restore() final;
        DLL void* GetNativeSurface() const final;
        DLL void PushGroup() final;
        DLL void PopGroup() final;
        DLL void ApplyDropShadow ( double aDx, double aDy,
                                   double aStdDeviationX, double aStdDeviationY,
                                   const Color& aFloodColor, double aFloodOpacity ) final;
        DLL uint8_t PickAtPoint ( double aX, double aY ) const final;
        DLL void ResetPick() final;
        DLL void SetClipRect ( double aX, double aY, double aWidth, double aHeight ) final;
        DLL std::unique_ptr<Path> CreatePath() const final;
    private:
        void InitSurfaces ( uint32_t aWidth, uint32_t aHeight );
        // Render surface
        sk_sp<SkSurface> mSurface;
        SkCanvas* mCanvas{};  // owned by mSurface
        // Pick surface (A8)
        std::vector<uint8_t> mPickPixels;
        uint32_t mWidth{0};
        uint32_t mHeight{0};
        // BGRA pixel cache (Skia stores RGBA, we expose BGRA for compatibility)
        mutable std::vector<uint8_t> mPixelCache;
        mutable bool mPixelCacheDirty{true};
        // Paint state
        ColorAttr mFillColor{};
        ColorAttr mStrokeColor{};
        double mStrokeWidth{1};
        double mStrokeOpacity{1};
        double mFillOpacity{1};
        double mOpacity{1};
        // Offscreen group stack
        struct SavedLayer
        {
            sk_sp<SkSurface> surface;
            SkCanvas* canvas;
        };
        std::vector<SavedLayer> mGroupStack;
    };
}
#endif
