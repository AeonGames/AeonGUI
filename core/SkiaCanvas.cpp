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

#include <core/SkSurface.h>
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#include <core/SkPath.h>
#include <core/SkPathBuilder.h>
#include <core/SkImage.h>
#include <core/SkData.h>
#include <core/SkMatrix.h>
#include <core/SkPathMeasure.h>
#include <core/SkBitmap.h>
#include <core/SkImageInfo.h>
#include <effects/SkImageFilters.h>
#if __has_include(<effects/SkGradientShader.h>)
#include <effects/SkGradientShader.h>
#define AEONGUI_HAS_SK_GRADIENT_SHADER 1
#else
#include <effects/SkGradient.h>
#define AEONGUI_HAS_SK_GRADIENT_SHADER 0
#endif
#include <pango/pango.h>
#include <pango/pangoft2.h>
#include <pango/pangofc-fontmap.h>
#include <hb.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "SkiaCanvas.hpp"
#include "SkiaPath.hpp"
#include "aeongui/FontDatabase.hpp"

namespace AeonGUI
{
    SkiaCanvas::SkiaCanvas() = default;

    SkiaCanvas::SkiaCanvas ( uint32_t aWidth, uint32_t aHeight )
    {
        InitSurfaces ( aWidth, aHeight );
    }

    void SkiaCanvas::InitSurfaces ( uint32_t aWidth, uint32_t aHeight )
    {
        mWidth = aWidth;
        mHeight = aHeight;
        SkImageInfo info = SkImageInfo::MakeN32Premul ( static_cast<int> ( aWidth ), static_cast<int> ( aHeight ) );
        mSurface = SkSurfaces::Raster ( info );
        mCanvas = mSurface ? mSurface->getCanvas() : nullptr;
        mPickPixels.assign ( static_cast<size_t> ( aWidth ) * aHeight, 0 );
        mPixelCache.resize ( static_cast<size_t> ( aWidth ) * aHeight * 4, 0 );
        mPixelCacheDirty = true;
    }

    SkiaCanvas::~SkiaCanvas() = default;

    void SkiaCanvas::ResizeViewport ( uint32_t aWidth, uint32_t aHeight )
    {
        if ( aWidth == mWidth && aHeight == mHeight )
        {
            return;
        }
        InitSurfaces ( aWidth, aHeight );
    }

    const uint8_t* SkiaCanvas::GetPixels() const
    {
        if ( !mSurface )
        {
            return nullptr;
        }
        if ( mPixelCacheDirty )
        {
            // Read pixels from Skia surface (RGBA premul) and convert to BGRA premul.
            SkImageInfo dstInfo = SkImageInfo::Make ( static_cast<int> ( mWidth ), static_cast<int> ( mHeight ),
                                  kBGRA_8888_SkColorType, kPremul_SkAlphaType );
            mSurface->readPixels ( dstInfo, mPixelCache.data(), mWidth * 4, 0, 0 );
            mPixelCacheDirty = false;
        }
        return mPixelCache.data();
    }

    size_t SkiaCanvas::GetWidth() const
    {
        return mWidth;
    }
    size_t SkiaCanvas::GetHeight() const
    {
        return mHeight;
    }
    size_t SkiaCanvas::GetStride() const
    {
        return static_cast<size_t> ( mWidth ) * 4;
    }

    void SkiaCanvas::Clear()
    {
        if ( mCanvas )
        {
            mCanvas->clear ( SK_ColorTRANSPARENT );
            mPixelCacheDirty = true;
        }
    }

    void SkiaCanvas::SetFillColor ( const ColorAttr& aColor )
    {
        mFillColor = aColor;
    }
    const ColorAttr& SkiaCanvas::GetFillColor() const
    {
        return mFillColor;
    }
    void SkiaCanvas::SetStrokeColor ( const ColorAttr& aColor )
    {
        mStrokeColor = aColor;
    }
    const ColorAttr& SkiaCanvas::GetStrokeColor() const
    {
        return mStrokeColor;
    }
    void SkiaCanvas::SetStrokeWidth ( double aWidth )
    {
        mStrokeWidth = aWidth;
    }
    double SkiaCanvas::GetStrokeWidth() const
    {
        return mStrokeWidth;
    }
    void SkiaCanvas::SetStrokeOpacity ( double aOpacity )
    {
        mStrokeOpacity = std::clamp ( aOpacity, 0.0, 1.0 );
    }
    double SkiaCanvas::GetStrokeOpacity() const
    {
        return mStrokeOpacity;
    }
    void SkiaCanvas::SetFillOpacity ( double aOpacity )
    {
        mFillOpacity = std::clamp ( aOpacity, 0.0, 1.0 );
    }
    double SkiaCanvas::GetFillOpacity() const
    {
        return mFillOpacity;
    }
    void SkiaCanvas::SetOpacity ( double aOpacity )
    {
        mOpacity = std::clamp ( aOpacity, 0.0, 1.0 );
    }
    double SkiaCanvas::GetOpacity() const
    {
        return mOpacity;
    }

    static hb_draw_funcs_t* GetHbDrawFuncs()
    {
        static hb_draw_funcs_t* funcs = []()
        {
            hb_draw_funcs_t* f = hb_draw_funcs_create();
            hb_draw_funcs_set_move_to_func ( f,
                                             [] ( hb_draw_funcs_t*, void* draw_data, hb_draw_state_t*,
                                                  float to_x, float to_y, void* )
            {
                static_cast<SkPathBuilder*> ( draw_data )->moveTo ( to_x, -to_y );
            }, nullptr, nullptr );
            hb_draw_funcs_set_line_to_func ( f,
                                             [] ( hb_draw_funcs_t*, void* draw_data, hb_draw_state_t*,
                                                  float to_x, float to_y, void* )
            {
                static_cast<SkPathBuilder*> ( draw_data )->lineTo ( to_x, -to_y );
            }, nullptr, nullptr );
            hb_draw_funcs_set_quadratic_to_func ( f,
                                                  [] ( hb_draw_funcs_t*, void* draw_data, hb_draw_state_t*,
                                                          float cx, float cy, float to_x, float to_y, void* )
            {
                static_cast<SkPathBuilder*> ( draw_data )->quadTo ( cx, -cy, to_x, -to_y );
            }, nullptr, nullptr );
            hb_draw_funcs_set_cubic_to_func ( f,
                                              [] ( hb_draw_funcs_t*, void* draw_data, hb_draw_state_t*,
                                                   float c1x, float c1y, float c2x, float c2y,
                                                   float to_x, float to_y, void* )
            {
                static_cast<SkPathBuilder*> ( draw_data )->cubicTo ( c1x, -c1y, c2x, -c2y, to_x, -to_y );
            }, nullptr, nullptr );
            hb_draw_funcs_set_close_path_func ( f,
                                                [] ( hb_draw_funcs_t*, void* draw_data, hb_draw_state_t*, void* )
            {
                static_cast<SkPathBuilder*> ( draw_data )->close();
            }, nullptr, nullptr );
            hb_draw_funcs_make_immutable ( f );
            return f;
        }
        ();
        return funcs;
    }

    static PangoFontDescription* CreateSkiaFontDescription ( const std::string& aFontFamily, double aFontSize,
            int aFontWeight, int aFontStyle )
    {
        PangoFontDescription* desc = pango_font_description_new();
        pango_font_description_set_family ( desc, aFontFamily.c_str() );
        pango_font_description_set_absolute_size ( desc, aFontSize * PANGO_SCALE );

        if ( aFontWeight <= 100 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_THIN );
        }
        else if ( aFontWeight <= 200 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_ULTRALIGHT );
        }
        else if ( aFontWeight <= 300 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_LIGHT );
        }
        else if ( aFontWeight <= 400 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_NORMAL );
        }
        else if ( aFontWeight <= 500 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_MEDIUM );
        }
        else if ( aFontWeight <= 600 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_SEMIBOLD );
        }
        else if ( aFontWeight <= 700 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_BOLD );
        }
        else if ( aFontWeight <= 800 )
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_ULTRABOLD );
        }
        else
        {
            pango_font_description_set_weight ( desc, PANGO_WEIGHT_HEAVY );
        }

        switch ( aFontStyle )
        {
        case 1:
            pango_font_description_set_style ( desc, PANGO_STYLE_ITALIC );
            break;
        case 2:
            pango_font_description_set_style ( desc, PANGO_STYLE_OBLIQUE );
            break;
        default:
            pango_font_description_set_style ( desc, PANGO_STYLE_NORMAL );
            break;
        }
        return desc;
    }

    static SkPath BuildTextPath ( PangoLayout* aLayout, double aBaselineY )
    {
        SkPathBuilder textPathBuilder;
        hb_draw_funcs_t* drawFuncs = GetHbDrawFuncs();
        PangoLayoutIter* iter = pango_layout_get_iter ( aLayout );
        do
        {
            PangoLayoutRun* run = pango_layout_iter_get_run_readonly ( iter );
            if ( !run )
            {
                continue;
            }
            PangoGlyphString* glyphs = run->glyphs;
            PangoFont* font = run->item->analysis.font;
            hb_font_t* hbFont = pango_font_get_hb_font ( font );
            if ( !hbFont )
            {
                continue;
            }

            PangoRectangle logical_rect;
            pango_layout_iter_get_run_extents ( iter, nullptr, &logical_rect );
            double runX = static_cast<double> ( logical_rect.x ) / PANGO_SCALE;

            double advanceX = 0.0;
            for ( int i = 0; i < glyphs->num_glyphs; ++i )
            {
                PangoGlyphInfo& gi = glyphs->glyphs[i];
                if ( gi.glyph == PANGO_GLYPH_EMPTY || ( gi.glyph & PANGO_GLYPH_UNKNOWN_FLAG ) )
                {
                    advanceX += static_cast<double> ( gi.geometry.width ) / PANGO_SCALE;
                    continue;
                }

                double xPos = runX + advanceX + static_cast<double> ( gi.geometry.x_offset ) / PANGO_SCALE;
                double yPos = aBaselineY + static_cast<double> ( gi.geometry.y_offset ) / PANGO_SCALE;

                SkPathBuilder glyphPathBuilder;
                hb_font_draw_glyph ( hbFont, gi.glyph, drawFuncs, &glyphPathBuilder );
                SkPath glyphPath = glyphPathBuilder.detach();
                if ( !glyphPath.isEmpty() )
                {
                    textPathBuilder.addPath ( glyphPath,
                                              static_cast<SkScalar> ( xPos ),
                                              static_cast<SkScalar> ( yPos ) );
                }

                advanceX += static_cast<double> ( gi.geometry.width ) / PANGO_SCALE;
            }
        }
        while ( pango_layout_iter_next_run ( iter ) );
        pango_layout_iter_free ( iter );
        return textPathBuilder.detach();
    }

    void SkiaCanvas::Draw ( const Path& aPath )
    {
        if ( !mCanvas )
        {
            return;
        }
        const SkiaPath& path = static_cast<const SkiaPath&> ( aPath );
        const SkPath& skPath = path.GetSkPath();

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            SkPaint layerPaint;
            layerPaint.setAlpha ( static_cast<U8CPU> ( mOpacity * 255.0 ) );
            mCanvas->saveLayer ( nullptr, &layerPaint );
        }

        // Fill
        if ( std::holds_alternative<Color> ( mFillColor ) )
        {
            const Color& fill = std::get<Color> ( mFillColor );
            SkPaint paint;
            paint.setStyle ( SkPaint::kFill_Style );
            paint.setAntiAlias ( true );
            double a = ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity;
            paint.setColor ( SkColorSetARGB (
                                 static_cast<U8CPU> ( a * 255.0 ),
                                 static_cast<U8CPU> ( fill.R() * 255.0 ),
                                 static_cast<U8CPU> ( fill.G() * 255.0 ),
                                 static_cast<U8CPU> ( fill.B() * 255.0 ) ) );
            mCanvas->drawPath ( skPath, paint );
        }
        else if ( std::holds_alternative<LinearGradient> ( mFillColor ) )
        {
            const LinearGradient& grad = std::get<LinearGradient> ( mFillColor );
            double gx1 = grad.x1, gy1 = grad.y1, gx2 = grad.x2, gy2 = grad.y2;
            if ( grad.objectBoundingBox )
            {
                SkRect bounds = skPath.getBounds();
                double bw = bounds.width();
                double bh = bounds.height();
                gx1 = bounds.fLeft + grad.x1 * bw;
                gy1 = bounds.fTop + grad.y1 * bh;
                gx2 = bounds.fLeft + grad.x2 * bw;
                gy2 = bounds.fTop + grad.y2 * bh;
            }
            std::vector<SkColor> colors;
            std::vector<float> positions;
            for ( const auto& stop : grad.stops )
            {
                double a = ( mFillOpacity >= 1.0 ) ? stop.color.A() : mFillOpacity;
                colors.push_back ( SkColorSetARGB (
                                       static_cast<U8CPU> ( a * 255.0 ),
                                       static_cast<U8CPU> ( stop.color.R() * 255.0 ),
                                       static_cast<U8CPU> ( stop.color.G() * 255.0 ),
                                       static_cast<U8CPU> ( stop.color.B() * 255.0 ) ) );
                positions.push_back ( static_cast<float> ( stop.offset ) );
            }
            SkPoint pts[2] = { SkPoint::Make ( static_cast<SkScalar> ( gx1 ), static_cast<SkScalar> ( gy1 ) ),
                               SkPoint::Make ( static_cast<SkScalar> ( gx2 ), static_cast<SkScalar> ( gy2 ) )
                             };
#if AEONGUI_HAS_SK_GRADIENT_SHADER
            sk_sp<SkShader> shader = SkGradientShader::MakeLinear (
                                         pts,
                                         colors.data(),
                                         positions.empty() ? nullptr : positions.data(),
                                         static_cast<int> ( colors.size() ),
                                         SkTileMode::kClamp );
#else
            std::vector<SkColor4f> colors4f;
            colors4f.reserve ( colors.size() );
            for ( const SkColor color : colors )
            {
                colors4f.push_back ( SkColor4f::FromColor ( color ) );
            }
            const SkGradient::Colors gradientColors (
                SkSpan<const SkColor4f> ( colors4f.data(), colors4f.size() ),
                SkSpan<const float> ( positions.data(), positions.size() ),
                SkTileMode::kClamp );
            const SkGradient gradient ( gradientColors, SkGradient::Interpolation::FromFlags ( 0 ) );
            sk_sp<SkShader> shader = SkShaders::LinearGradient ( pts, gradient );
#endif
            SkPaint paint;
            paint.setStyle ( SkPaint::kFill_Style );
            paint.setAntiAlias ( true );
            paint.setShader ( shader );
            mCanvas->drawPath ( skPath, paint );
        }

        // Stroke
        if ( std::holds_alternative<Color> ( mStrokeColor ) )
        {
            const Color& stroke = std::get<Color> ( mStrokeColor );
            SkPaint paint;
            paint.setStyle ( SkPaint::kStroke_Style );
            paint.setAntiAlias ( true );
            paint.setStrokeWidth ( static_cast<SkScalar> ( mStrokeWidth ) );
            double a = ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity;
            paint.setColor ( SkColorSetARGB (
                                 static_cast<U8CPU> ( a * 255.0 ),
                                 static_cast<U8CPU> ( stroke.R() * 255.0 ),
                                 static_cast<U8CPU> ( stroke.G() * 255.0 ),
                                 static_cast<U8CPU> ( stroke.B() * 255.0 ) ) );
            mCanvas->drawPath ( skPath, paint );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            mCanvas->restore();
        }

        // Compute device-space bounding box for this pick ID.
        if ( mPickId > 0 )
        {
            SkRect bounds;
            if ( std::holds_alternative<Color> ( mStrokeColor ) )
            {
                bounds = skPath.getBounds();
                // Expand by half stroke width.
                SkScalar half = static_cast<SkScalar> ( mStrokeWidth * 0.5 );
                bounds.outset ( half, half );
            }
            else
            {
                bounds = skPath.getBounds();
            }
            // Transform corners to device space.
            SkMatrix ctm = mCanvas->getTotalMatrix();
            SkRect devBounds;
            ctm.mapRect ( &devBounds, bounds );
            mPickBounds[mPickId] = { static_cast<double> ( devBounds.fLeft ),
                                     static_cast<double> ( devBounds.fTop ),
                                     static_cast<double> ( devBounds.fRight ),
                                     static_cast<double> ( devBounds.fBottom )
                                   };
        }

        // Fill pick buffer
        if ( mPickId > 0 && mWidth > 0 && mHeight > 0 )
        {
            // Create a temporary A8 surface to rasterize the pick shape.
            SkImageInfo a8Info = SkImageInfo::MakeA8 ( static_cast<int> ( mWidth ), static_cast<int> ( mHeight ) );
            auto pickSurf = SkSurfaces::Raster ( a8Info );
            if ( pickSurf )
            {
                SkCanvas* pc = pickSurf->getCanvas();
                pc->clear ( SK_ColorTRANSPARENT );
                // Copy the current transform so the path is rendered in the same space.
                pc->setMatrix ( mCanvas->getTotalMatrix() );
                SkPaint pickPaint;
                pickPaint.setStyle ( SkPaint::kFill_Style );
                pickPaint.setAntiAlias ( false );
                pickPaint.setColor ( SkColorSetARGB ( mPickId, 0, 0, 0 ) );
                pc->drawPath ( skPath, pickPaint );
                // Read back A8 and merge into our pick buffer.
                SkPixmap pm;
                if ( pickSurf->peekPixels ( &pm ) )
                {
                    for ( uint32_t y = 0; y < mHeight; ++y )
                    {
                        const uint8_t* row = static_cast<const uint8_t*> ( pm.addr() ) + y * pm.rowBytes();
                        uint8_t* dst = mPickPixels.data() + y * mWidth;
                        for ( uint32_t x = 0; x < mWidth; ++x )
                        {
                            if ( row[x] > 0 )
                            {
                                dst[x] = mPickId;
                            }
                        }
                    }
                }
            }
        }
        mPixelCacheDirty = true;
    }

    void SkiaCanvas::DrawImage ( const uint8_t* aPixels,
                                 size_t aImageWidth,
                                 size_t aImageHeight,
                                 size_t aImageStride,
                                 double aX,
                                 double aY,
                                 double aWidth,
                                 double aHeight,
                                 double aOpacity )
    {
        if ( !mCanvas || !aPixels || aImageWidth == 0 || aImageHeight == 0 || aWidth <= 0 || aHeight <= 0 )
        {
            return;
        }
        double opacity = std::clamp ( aOpacity, 0.0, 1.0 );
        if ( opacity <= 0.0 )
        {
            return;
        }

        // Source is BGRA; wrap as SkImage in BGRA format.
        SkImageInfo srcInfo = SkImageInfo::Make ( static_cast<int> ( aImageWidth ),
                              static_cast<int> ( aImageHeight ),
                              kBGRA_8888_SkColorType, kPremul_SkAlphaType );
        SkBitmap bmp;
        bmp.installPixels ( srcInfo, const_cast<uint8_t*> ( aPixels ), aImageStride );
        sk_sp<SkImage> image = bmp.asImage();
        if ( !image )
        {
            return;
        }

        SkPaint paint;
        paint.setAlpha ( static_cast<U8CPU> ( opacity * 255.0 ) );
        SkRect dst = SkRect::MakeXYWH ( static_cast<SkScalar> ( aX ), static_cast<SkScalar> ( aY ),
                                        static_cast<SkScalar> ( aWidth ), static_cast<SkScalar> ( aHeight ) );
        mCanvas->drawImageRect ( image, dst, SkSamplingOptions ( SkFilterMode::kLinear ), &paint );
        mPixelCacheDirty = true;
    }

    void SkiaCanvas::DrawText ( const std::string& aText, double aX, double aY,
                                const std::string& aFontFamily, double aFontSize,
                                int aFontWeight, int aFontStyle )
    {
        if ( aText.empty() || !mCanvas )
        {
            return;
        }

        if ( !FontDatabase::GetFontMap() )
        {
            return;
        }
        PangoContext* pangoContext = FontDatabase::CreateContext();
        PangoLayout* layout = pango_layout_new ( pangoContext );
        PangoFontDescription* desc = CreateSkiaFontDescription ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        pango_layout_set_font_description ( layout, desc );
        pango_layout_set_text ( layout, aText.c_str(), -1 );

        PangoLayoutIter* baseIter = pango_layout_get_iter ( layout );
        int baseline = pango_layout_iter_get_baseline ( baseIter );
        pango_layout_iter_free ( baseIter );
        ( void ) baseline;

        // aY is the SVG baseline. FreeType glyph outlines have their origin at
        // the baseline, so pass aY directly — not the top-of-layout offset.
        SkPath textPath = BuildTextPath ( layout, aY );
        textPath = textPath.makeOffset ( static_cast<SkScalar> ( aX ), 0 );

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            SkPaint layerPaint;
            layerPaint.setAlpha ( static_cast<U8CPU> ( mOpacity * 255.0 ) );
            mCanvas->saveLayer ( nullptr, &layerPaint );
        }

        // Fill
        if ( std::holds_alternative<Color> ( mFillColor ) )
        {
            const Color& fill = std::get<Color> ( mFillColor );
            SkPaint paint;
            double a = ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity;
            paint.setColor ( SkColorSetARGB (
                                 static_cast<U8CPU> ( a * 255.0 ),
                                 static_cast<U8CPU> ( fill.R() * 255.0 ),
                                 static_cast<U8CPU> ( fill.G() * 255.0 ),
                                 static_cast<U8CPU> ( fill.B() * 255.0 ) ) );
            paint.setAntiAlias ( true );
            mCanvas->drawPath ( textPath, paint );
        }

        // Stroke
        if ( std::holds_alternative<Color> ( mStrokeColor ) )
        {
            const Color& stroke = std::get<Color> ( mStrokeColor );
            SkPaint paint;
            paint.setStyle ( SkPaint::kStroke_Style );
            paint.setStrokeWidth ( static_cast<SkScalar> ( mStrokeWidth ) );
            double a = ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity;
            paint.setColor ( SkColorSetARGB (
                                 static_cast<U8CPU> ( a * 255.0 ),
                                 static_cast<U8CPU> ( stroke.R() * 255.0 ),
                                 static_cast<U8CPU> ( stroke.G() * 255.0 ),
                                 static_cast<U8CPU> ( stroke.B() * 255.0 ) ) );
            paint.setAntiAlias ( true );
            mCanvas->drawPath ( textPath, paint );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            mCanvas->restore();
        }
        mPixelCacheDirty = true;

        pango_font_description_free ( desc );
        g_object_unref ( layout );
        g_object_unref ( pangoContext );
    }

    double SkiaCanvas::MeasureText ( const std::string& aText,
                                     const std::string& aFontFamily, double aFontSize,
                                     int aFontWeight, int aFontStyle ) const
    {
        if ( aText.empty() )
        {
            return 0.0;
        }

        if ( !FontDatabase::GetFontMap() )
        {
            return 0.0;
        }
        PangoContext* pangoContext = FontDatabase::CreateContext();
        PangoLayout* layout = pango_layout_new ( pangoContext );
        PangoFontDescription* desc = CreateSkiaFontDescription ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        pango_layout_set_font_description ( layout, desc );
        pango_layout_set_text ( layout, aText.c_str(), -1 );

        int width = 0;
        int height = 0;
        pango_layout_get_pixel_size ( layout, &width, &height );

        pango_font_description_free ( desc );
        g_object_unref ( layout );
        g_object_unref ( pangoContext );

        return static_cast<double> ( width );
    }

    void SkiaCanvas::DrawTextOnPath ( const std::string& aText,
                                      const Path& aPath,
                                      double aStartOffset,
                                      const std::string& aFontFamily, double aFontSize,
                                      int aFontWeight, int aFontStyle,
                                      bool aReverse, bool aClosed )
    {
        if ( aText.empty() || !mCanvas )
        {
            return;
        }

        if ( !FontDatabase::GetFontMap() )
        {
            return;
        }
        PangoContext* pangoContext = FontDatabase::CreateContext();
        PangoLayout* layout = pango_layout_new ( pangoContext );
        PangoFontDescription* desc = CreateSkiaFontDescription ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        pango_layout_set_font_description ( layout, desc );

        // Get baseline offset for vertical centering.
        pango_layout_set_text ( layout, aText.c_str(), -1 );
        PangoLayoutIter* baseIter = pango_layout_get_iter ( layout );
        int baseline = pango_layout_iter_get_baseline ( baseIter );
        pango_layout_iter_free ( baseIter );
        double baselineOffset = static_cast<double> ( baseline ) / PANGO_SCALE;

        double pathLength = aPath.GetTotalLength();

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            SkPaint layerPaint;
            layerPaint.setAlpha ( static_cast<U8CPU> ( mOpacity * 255.0 ) );
            mCanvas->saveLayer ( nullptr, &layerPaint );
        }

        // Render character-by-character along the path.
        double distance = aStartOffset;
        size_t pos = 0;
        while ( pos < aText.size() )
        {
            unsigned char lead = static_cast<unsigned char> ( aText[pos] );
            int charLen = 1;
            if ( lead >= 0xF0 )
            {
                charLen = 4;
            }
            else if ( lead >= 0xE0 )
            {
                charLen = 3;
            }
            else if ( lead >= 0xC0 )
            {
                charLen = 2;
            }
            if ( pos + static_cast<size_t> ( charLen ) > aText.size() )
            {
                break;
            }
            std::string oneChar = aText.substr ( pos, static_cast<size_t> ( charLen ) );

            // Measure this character's advance width.
            pango_layout_set_text ( layout, oneChar.c_str(), charLen );
            int charWidth = 0, charHeight = 0;
            pango_layout_get_pixel_size ( layout, &charWidth, &charHeight );
            double advance = static_cast<double> ( charWidth );

            double mid = distance + advance * 0.5;
            double queryMid = aReverse ? ( pathLength - mid ) : mid;

            if ( aClosed && pathLength > 0.0 )
            {
                queryMid = std::fmod ( queryMid, pathLength );
                if ( queryMid < 0.0 )
                {
                    queryMid += pathLength;
                }
            }
            else
            {
                if ( queryMid < 0.0 || queryMid > pathLength )
                {
                    distance += advance;
                    pos += static_cast<size_t> ( charLen );
                    continue;
                }
            }

            PathPoint pt = aPath.GetPointAtLength ( queryMid );
            double angle = aReverse ? ( pt.angle + M_PI ) : pt.angle;

            // Build glyph outline for this character.
            SkPath charPath = BuildTextPath ( layout, 0.0 );

            if ( !charPath.isEmpty() )
            {
                mCanvas->save();
                mCanvas->translate ( static_cast<SkScalar> ( pt.x ), static_cast<SkScalar> ( pt.y ) );
                mCanvas->rotate ( static_cast<SkScalar> ( angle * 180.0 / M_PI ) );
                mCanvas->translate ( static_cast<SkScalar> ( -advance * 0.5 ), static_cast<SkScalar> ( -baselineOffset ) );

                if ( std::holds_alternative<Color> ( mFillColor ) )
                {
                    const Color& fill = std::get<Color> ( mFillColor );
                    SkPaint paint;
                    double a = ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity;
                    paint.setColor ( SkColorSetARGB (
                                         static_cast<U8CPU> ( a * 255.0 ),
                                         static_cast<U8CPU> ( fill.R() * 255.0 ),
                                         static_cast<U8CPU> ( fill.G() * 255.0 ),
                                         static_cast<U8CPU> ( fill.B() * 255.0 ) ) );
                    paint.setAntiAlias ( true );
                    mCanvas->drawPath ( charPath, paint );
                }
                if ( std::holds_alternative<Color> ( mStrokeColor ) )
                {
                    const Color& stroke = std::get<Color> ( mStrokeColor );
                    SkPaint paint;
                    paint.setStyle ( SkPaint::kStroke_Style );
                    paint.setStrokeWidth ( static_cast<SkScalar> ( mStrokeWidth ) );
                    double a = ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity;
                    paint.setColor ( SkColorSetARGB (
                                         static_cast<U8CPU> ( a * 255.0 ),
                                         static_cast<U8CPU> ( stroke.R() * 255.0 ),
                                         static_cast<U8CPU> ( stroke.G() * 255.0 ),
                                         static_cast<U8CPU> ( stroke.B() * 255.0 ) ) );
                    paint.setAntiAlias ( true );
                    mCanvas->drawPath ( charPath, paint );
                }

                mCanvas->restore();
            }

            distance += advance;
            pos += static_cast<size_t> ( charLen );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            mCanvas->restore();
        }
        mPixelCacheDirty = true;

        pango_font_description_free ( desc );
        g_object_unref ( layout );
        g_object_unref ( pangoContext );
    }

    void SkiaCanvas::SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio )
    {
        if ( !mCanvas )
        {
            return;
        }
        double scale_x = GetWidth() / aViewBox.width;
        double scale_y = GetHeight() / aViewBox.height;
        if ( aPreserveAspectRatio.GetAlign() != PreserveAspectRatio::Align::none )
        {
            if ( aPreserveAspectRatio.GetMeetOrSlice() == PreserveAspectRatio::MeetOrSlice::Meet )
            {
                scale_x = std::min ( scale_x, scale_y );
                scale_y = scale_x;
            }
            else if ( aPreserveAspectRatio.GetMeetOrSlice() == PreserveAspectRatio::MeetOrSlice::Slice )
            {
                scale_x = std::max ( scale_x, scale_y );
                scale_y = scale_x;
            }
        }
        double translate_x = -aViewBox.min_x * scale_x;
        double translate_y = -aViewBox.min_y * scale_y;

        auto x_align = aPreserveAspectRatio.GetAlignX();
        auto y_align = aPreserveAspectRatio.GetAlignY();

        if ( x_align == PreserveAspectRatio::MinMidMax::Mid )
        {
            translate_x += ( GetWidth() - aViewBox.width * scale_x ) / 2;
        }
        else if ( x_align == PreserveAspectRatio::MinMidMax::Max )
        {
            translate_x += ( GetWidth() - aViewBox.width * scale_x );
        }
        if ( y_align == PreserveAspectRatio::MinMidMax::Mid )
        {
            translate_y += ( GetHeight() - aViewBox.height * scale_y ) / 2;
        }
        else if ( y_align == PreserveAspectRatio::MinMidMax::Max )
        {
            translate_y += ( GetHeight() - aViewBox.height * scale_y );
        }

        SkMatrix m;
        m.setAll ( static_cast<SkScalar> ( scale_x ), 0, static_cast<SkScalar> ( translate_x ),
                   0, static_cast<SkScalar> ( scale_y ), static_cast<SkScalar> ( translate_y ),
                   0, 0, 1 );
        mCanvas->setMatrix ( m );
    }

    void SkiaCanvas::SetTransform ( const Matrix2x3& aMatrix )
    {
        if ( !mCanvas )
        {
            return;
        }
        SkMatrix m;
        m.setAll ( static_cast<SkScalar> ( aMatrix[0] ), static_cast<SkScalar> ( aMatrix[2] ), static_cast<SkScalar> ( aMatrix[4] ),
                   static_cast<SkScalar> ( aMatrix[1] ), static_cast<SkScalar> ( aMatrix[3] ), static_cast<SkScalar> ( aMatrix[5] ),
                   0, 0, 1 );
        mCanvas->setMatrix ( m );
    }

    void SkiaCanvas::Transform ( const Matrix2x3& aMatrix )
    {
        if ( !mCanvas )
        {
            return;
        }
        SkMatrix m;
        m.setAll ( static_cast<SkScalar> ( aMatrix[0] ), static_cast<SkScalar> ( aMatrix[2] ), static_cast<SkScalar> ( aMatrix[4] ),
                   static_cast<SkScalar> ( aMatrix[1] ), static_cast<SkScalar> ( aMatrix[3] ), static_cast<SkScalar> ( aMatrix[5] ),
                   0, 0, 1 );
        mCanvas->concat ( m );
    }

    void SkiaCanvas::Save()
    {
        if ( mCanvas )
        {
            mCanvas->save();
        }
    }

    void SkiaCanvas::Restore()
    {
        if ( mCanvas )
        {
            mCanvas->restore();
        }
    }

    void* SkiaCanvas::GetNativeSurface() const
    {
        return mSurface.get();
    }

    void SkiaCanvas::PushGroup()
    {
        if ( !mCanvas )
        {
            return;
        }
        // Create offscreen surface matching current dimensions.
        SkImageInfo info = SkImageInfo::MakeN32Premul ( static_cast<int> ( mWidth ), static_cast<int> ( mHeight ) );
        auto offscreen = SkSurfaces::Raster ( info );
        if ( offscreen )
        {
            // Copy the current matrix to the offscreen canvas.
            offscreen->getCanvas()->setMatrix ( mCanvas->getTotalMatrix() );
            mGroupStack.push_back ( {mSurface, mCanvas} );
            mSurface = offscreen;
            mCanvas = mSurface->getCanvas();
        }
    }

    void SkiaCanvas::PopGroup()
    {
        if ( mGroupStack.empty() )
        {
            return;
        }
        // Capture current offscreen as image.
        sk_sp<SkImage> image = mSurface->makeImageSnapshot();
        // Restore parent surface.
        auto saved = std::move ( mGroupStack.back() );
        mGroupStack.pop_back();
        mSurface = saved.surface;
        mCanvas = saved.canvas;
        // Draw offscreen result onto parent.
        if ( image )
        {
            mCanvas->save();
            mCanvas->resetMatrix();
            mCanvas->drawImage ( image, 0, 0 );
            mCanvas->restore();
        }
        mPixelCacheDirty = true;
    }

    void SkiaCanvas::ApplyDropShadow ( double aDx, double aDy,
                                       double aStdDeviationX, double aStdDeviationY,
                                       const Color& aFloodColor, double aFloodOpacity )
    {
        if ( mGroupStack.empty() )
        {
            return;
        }
        // Capture content from the current (offscreen) group.
        sk_sp<SkImage> contentImage = mSurface->makeImageSnapshot();
        // Restore parent.
        auto saved = std::move ( mGroupStack.back() );
        mGroupStack.pop_back();
        mSurface = saved.surface;
        mCanvas = saved.canvas;

        if ( !contentImage )
        {
            return;
        }

        double opacity = std::clamp ( aFloodOpacity, 0.0, 1.0 );

        // Draw shadow: use the content alpha as mask, flood-colored and blurred.
        {
            SkPaint shadowPaint;
            // Color filter to replace RGB with flood color, keeping alpha.
            SkColor floodSk = SkColorSetARGB (
                                  static_cast<U8CPU> ( opacity * 255.0 ),
                                  static_cast<U8CPU> ( aFloodColor.R() * 255.0 ),
                                  static_cast<U8CPU> ( aFloodColor.G() * 255.0 ),
                                  static_cast<U8CPU> ( aFloodColor.B() * 255.0 ) );
            // Use DropShadowOnly to get just the shadow.
            sk_sp<SkImageFilter> dropShadow = SkImageFilters::DropShadowOnly (
                                                  static_cast<SkScalar> ( aDx ), static_cast<SkScalar> ( aDy ),
                                                  static_cast<SkScalar> ( aStdDeviationX ), static_cast<SkScalar> ( aStdDeviationY ),
                                                  floodSk, nullptr );
            shadowPaint.setImageFilter ( dropShadow );
            mCanvas->save();
            mCanvas->resetMatrix();
            mCanvas->drawImage ( contentImage, 0, 0, SkSamplingOptions(), &shadowPaint );
            mCanvas->restore();
        }

        // Draw content on top.
        mCanvas->save();
        mCanvas->resetMatrix();
        mCanvas->drawImage ( contentImage, 0, 0 );
        mCanvas->restore();
        mPixelCacheDirty = true;
    }

    uint8_t SkiaCanvas::PickAtPoint ( double aX, double aY ) const
    {
        int ix = static_cast<int> ( aX );
        int iy = static_cast<int> ( aY );
        if ( ix < 0 || iy < 0 || static_cast<uint32_t> ( ix ) >= mWidth || static_cast<uint32_t> ( iy ) >= mHeight )
        {
            return 0;
        }
        return mPickPixels[iy * mWidth + ix];
    }

    void SkiaCanvas::ResetPick()
    {
        mPickId = 0;
        std::fill ( mPickPixels.begin(), mPickPixels.end(), static_cast<uint8_t> ( 0 ) );
    }

    void SkiaCanvas::SetClipRect ( double aX, double aY, double aWidth, double aHeight )
    {
        if ( !mCanvas )
        {
            return;
        }
        // Clip in device (pixel) coordinates.
        mCanvas->save();
        mCanvas->resetMatrix();
        mCanvas->clipRect ( SkRect::MakeXYWH ( static_cast<SkScalar> ( aX ), static_cast<SkScalar> ( aY ),
                                               static_cast<SkScalar> ( aWidth ), static_cast<SkScalar> ( aHeight ) ) );
        mCanvas->restore();
    }

    std::unique_ptr<Path> SkiaCanvas::CreatePath() const
    {
        return std::make_unique<SkiaPath>();
    }
}
