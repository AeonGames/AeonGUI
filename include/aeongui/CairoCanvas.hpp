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
        DLL CairoCanvas ();
        /** @brief Construct a canvas with the given viewport size.
         *  @param aWidth  Initial width in pixels.
         *  @param aHeight Initial height in pixels.
         */
        DLL CairoCanvas ( uint32_t aWidth, uint32_t aHeight );
        /** @brief Destructor. Releases Cairo resources. */
        DLL ~CairoCanvas() final;
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
        DLL void* GetNativeSurface() const final;
    private:
        cairo_surface_t* mCairoSurface{};
        cairo_t* mCairoContext{};
        ColorAttr mFillColor{};
        ColorAttr mStrokeColor{};
        double mStrokeWidth{1};
        double mStrokeOpacity{1};
        double mFillOpacity{1};
        double mOpacity{1};
    };
}
#endif
