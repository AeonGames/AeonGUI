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
#ifndef AEONGUI_CAIROCANVAS_H
#define AEONGUI_CAIROCANVAS_H
#include <cstdint>
#include <string>
#include "aeongui/Canvas.hpp"

struct _cairo_surface;
typedef struct _cairo_surface cairo_surface_t;
struct _cairo;
typedef struct _cairo cairo_t;

namespace AeonGUI
{
    /** @brief Cairo-backed Canvas implementation.
     *
     *  Provides software rasterization of paths, images, and text
     *  using the Cairo 2D graphics library.
     */
    class CairoCanvas : public Canvas
    {
    public:
        /** @brief Default constructor. Creates an empty canvas. */
        AEONGUI_DLL CairoCanvas ();
        /** @brief Construct a canvas with the given viewport size.
         *  @param aWidth  Initial width in pixels.
         *  @param aHeight Initial height in pixels.
         */
        AEONGUI_DLL CairoCanvas ( uint32_t aWidth, uint32_t aHeight );
        /** @brief Destructor. Releases Cairo resources. */
        AEONGUI_DLL ~CairoCanvas() final;
        AEONGUI_DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) final;
        AEONGUI_DLL const uint8_t* GetPixels() const final;
        AEONGUI_DLL size_t GetWidth() const final;
        AEONGUI_DLL size_t GetHeight() const final;
        AEONGUI_DLL size_t GetStride() const final;
        AEONGUI_DLL void Clear() final;
        AEONGUI_DLL void Draw ( const Path& aPath ) final;
        AEONGUI_DLL void DrawImage ( const uint8_t* aPixels,
                                     size_t aImageWidth,
                                     size_t aImageHeight,
                                     size_t aImageStride,
                                     double aX,
                                     double aY,
                                     double aWidth,
                                     double aHeight,
                                     double aOpacity ) final;
        AEONGUI_DLL void DrawText ( const std::string& aText, double aX, double aY,
                                    const std::string& aFontFamily, double aFontSize,
                                    int aFontWeight, int aFontStyle ) final;
        AEONGUI_DLL double MeasureText ( const std::string& aText,
                                         const std::string& aFontFamily, double aFontSize,
                                         int aFontWeight, int aFontStyle ) const final;
        AEONGUI_DLL void DrawTextOnPath ( const std::string& aText,
                                          const Path& aPath,
                                          double aStartOffset,
                                          const std::string& aFontFamily, double aFontSize,
                                          int aFontWeight, int aFontStyle,
                                          bool aReverse = false, bool aClosed = false ) final;
        AEONGUI_DLL void SetFillColor ( const ColorAttr& aColor ) final;
        AEONGUI_DLL const ColorAttr& GetFillColor() const final;
        AEONGUI_DLL void SetStrokeColor ( const ColorAttr& aColor ) final;
        AEONGUI_DLL const ColorAttr& GetStrokeColor() const final;
        AEONGUI_DLL void SetStrokeWidth ( double aWidth ) final;
        AEONGUI_DLL double GetStrokeWidth () const final;
        AEONGUI_DLL void SetStrokeOpacity ( double aWidth ) final;
        AEONGUI_DLL double GetStrokeOpacity () const final;
        AEONGUI_DLL void SetFillOpacity ( double aWidth ) final;
        AEONGUI_DLL double GetFillOpacity () const final;
        AEONGUI_DLL void SetOpacity ( double aWidth ) final;
        AEONGUI_DLL double GetOpacity () const final;
        AEONGUI_DLL void SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio ) final;
        AEONGUI_DLL void SetTransform ( const Matrix2x3& aMatrix ) final;
        AEONGUI_DLL void Transform ( const Matrix2x3& aMatrix ) final;
        AEONGUI_DLL void Save() final;
        AEONGUI_DLL void Restore() final;
        AEONGUI_DLL void* GetNativeSurface() const final;
        AEONGUI_DLL void PushGroup() final;
        AEONGUI_DLL void PopGroup() final;
        AEONGUI_DLL void ApplyDropShadow ( double aDx, double aDy,
                                           double aStdDeviationX, double aStdDeviationY,
                                           const Color& aFloodColor, double aFloodOpacity ) final;
        AEONGUI_DLL uint8_t PickAtPoint ( double aX, double aY ) const final;
        AEONGUI_DLL void ResetPick() final;
        AEONGUI_DLL void SetClipRect ( double aX, double aY, double aWidth, double aHeight ) final;
        AEONGUI_DLL std::unique_ptr<Path> CreatePath() const final;
    private:
        void InitPickSurface ( uint32_t aWidth, uint32_t aHeight );
        void DestroyPickSurface();
        cairo_surface_t* mCairoSurface{};
        cairo_t* mCairoContext{};
        cairo_surface_t* mPickSurface{};
        cairo_t* mPickContext{};
        ColorAttr mFillColor{};
        ColorAttr mStrokeColor{};
        double mStrokeWidth{1};
        double mStrokeOpacity{1};
        double mFillOpacity{1};
        double mOpacity{1};
    };
}
#endif
