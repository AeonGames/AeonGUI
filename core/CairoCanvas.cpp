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

#include <cairo.h>
#include <pango/pango.h>
#include <pango/pangocairo.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "CairoCanvas.hpp"
#include "CairoPath.hpp"
#include "PangoTextLayout.hpp"
#include "aeongui/FontDatabase.hpp"

namespace AeonGUI
{
    CairoCanvas::CairoCanvas () = default;
    CairoCanvas::CairoCanvas ( uint32_t aWidth, uint32_t aHeight ) :
        mCairoSurface{cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight ) },
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) }
    {
        InitPickSurface ( aWidth, aHeight );
    }

    void CairoCanvas::ResizeViewport ( uint32_t aWidth, uint32_t aHeight )
    {
        if ( aWidth == GetWidth() && aHeight == GetHeight() )
        {
            return;
        }
        if ( mCairoContext )
        {
            cairo_destroy ( mCairoContext );
        }
        if ( mCairoSurface )
        {
            cairo_surface_destroy ( mCairoSurface );
        }
        mCairoSurface = cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight );
        mCairoContext = cairo_create ( mCairoSurface );
        DestroyPickSurface();
        InitPickSurface ( aWidth, aHeight );
    }

    const uint8_t* CairoCanvas::GetPixels() const
    {
        if ( mCairoSurface )
        {
            cairo_surface_flush ( const_cast<cairo_surface_t*> ( mCairoSurface ) );
        }
        return cairo_image_surface_get_data ( mCairoSurface );
    }

    size_t CairoCanvas::GetWidth() const
    {
        return mCairoSurface ? static_cast<size_t> ( cairo_image_surface_get_width ( mCairoSurface ) ) : 0;
    }

    size_t CairoCanvas::GetHeight() const
    {
        return mCairoSurface ? static_cast<size_t> ( cairo_image_surface_get_height ( mCairoSurface ) ) : 0;
    }

    size_t CairoCanvas::GetStride() const
    {
        return mCairoSurface ? static_cast<size_t> ( cairo_image_surface_get_stride ( mCairoSurface ) ) : 0;
    }

    void CairoCanvas::Clear()
    {
        cairo_save ( mCairoContext );
        cairo_set_operator ( mCairoContext, CAIRO_OPERATOR_CLEAR );
        cairo_paint ( mCairoContext );
        cairo_restore ( mCairoContext );
    }
    CairoCanvas::~CairoCanvas()
    {
        DestroyPickSurface();
        if ( mCairoContext )
        {
            cairo_destroy ( mCairoContext );
        }
        if ( mCairoSurface )
        {
            cairo_surface_destroy ( mCairoSurface );
        }
    }

    void CairoCanvas::SetFillColor ( const ColorAttr& aColor )
    {
        mFillColor = aColor;
    }

    const ColorAttr& CairoCanvas::GetFillColor() const
    {
        return mFillColor;
    }

    void CairoCanvas::SetStrokeColor ( const ColorAttr& aColor )
    {
        mStrokeColor = aColor;
    }

    const ColorAttr& CairoCanvas::GetStrokeColor() const
    {
        return mStrokeColor;
    }

    void CairoCanvas::SetStrokeWidth ( double aStrokeWidth )
    {
        mStrokeWidth = aStrokeWidth;
    }

    double CairoCanvas::GetStrokeWidth () const
    {
        return mStrokeWidth;
    }

    void CairoCanvas::SetStrokeOpacity ( double aStrokeOpacity )
    {
        mStrokeOpacity = ( ( aStrokeOpacity < 0.0 ) ? 0.0 : ( aStrokeOpacity > 1.0 ) ? 1.0 : aStrokeOpacity );
    }

    double CairoCanvas::GetStrokeOpacity () const
    {
        return mStrokeOpacity;
    }

    void CairoCanvas::SetFillOpacity ( double aFillOpacity )
    {
        mFillOpacity = ( ( aFillOpacity < 0.0 ) ? 0.0 : ( aFillOpacity > 1.0 ) ? 1.0 : aFillOpacity );
    }

    double CairoCanvas::GetFillOpacity () const
    {
        return mFillOpacity;
    }

    void CairoCanvas::SetOpacity ( double aOpacity )
    {
        mOpacity = ( ( aOpacity < 0.0 ) ? 0.0 : ( aOpacity > 1.0 ) ? 1.0 : aOpacity );
    }

    double CairoCanvas::GetOpacity () const
    {
        return mOpacity;
    }

    void CairoCanvas::Draw ( const Path& aPath )
    {
        const CairoPath& path = reinterpret_cast<const CairoPath&> ( aPath );
        cairo_append_path ( mCairoContext, path.GetCairoPath() );
        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_push_group ( mCairoContext );
        }
        if ( std::holds_alternative<Color> ( mFillColor ) )
        {
            Color& fill = std::get<Color> ( mFillColor );
            cairo_set_source_rgba ( mCairoContext, fill.R(), fill.G(), fill.B(), ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity );
            cairo_fill_preserve ( mCairoContext );
        }
        else if ( std::holds_alternative<LinearGradient> ( mFillColor ) )
        {
            const LinearGradient& grad = std::get<LinearGradient> ( mFillColor );
            double gx1 = grad.x1, gy1 = grad.y1, gx2 = grad.x2, gy2 = grad.y2;
            if ( grad.objectBoundingBox )
            {
                double bx1, by1, bx2, by2;
                cairo_path_extents ( mCairoContext, &bx1, &by1, &bx2, &by2 );
                double bw = bx2 - bx1;
                double bh = by2 - by1;
                gx1 = bx1 + grad.x1 * bw;
                gy1 = by1 + grad.y1 * bh;
                gx2 = bx1 + grad.x2 * bw;
                gy2 = by1 + grad.y2 * bh;
            }
            cairo_pattern_t* pattern = cairo_pattern_create_linear ( gx1, gy1, gx2, gy2 );
            for ( const auto& stop : grad.stops )
            {
                cairo_pattern_add_color_stop_rgba ( pattern, stop.offset,
                                                    stop.color.R(), stop.color.G(), stop.color.B(),
                                                    ( mFillOpacity >= 1.0 ) ? stop.color.A() : mFillOpacity );
            }
            cairo_set_source ( mCairoContext, pattern );
            cairo_fill_preserve ( mCairoContext );
            cairo_pattern_destroy ( pattern );
        }
        if ( std::holds_alternative<Color> ( mStrokeColor ) )
        {
            Color& stroke = std::get<Color> ( mStrokeColor );
            cairo_set_line_width ( mCairoContext, mStrokeWidth );
            cairo_set_source_rgba ( mCairoContext, stroke.R(), stroke.G(), stroke.B(), ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity );
            cairo_stroke_preserve ( mCairoContext );
        }
        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_pop_group_to_source ( mCairoContext );
            cairo_paint_with_alpha ( mCairoContext, mOpacity );
        }
        // Compute device-space bounding box for this pick ID.
        if ( mPickId > 0 )
        {
            double ux1, uy1, ux2, uy2;
            if ( std::holds_alternative<Color> ( mStrokeColor ) )
            {
                cairo_stroke_extents ( mCairoContext, &ux1, &uy1, &ux2, &uy2 );
            }
            else
            {
                cairo_path_extents ( mCairoContext, &ux1, &uy1, &ux2, &uy2 );
            }
            // Transform all four corners to device space for correct AABB under rotation.
            double cx[4] = { ux1, ux2, ux2, ux1 };
            double cy[4] = { uy1, uy1, uy2, uy2 };
            double dx1 = 1e30, dy1 = 1e30, dx2 = -1e30, dy2 = -1e30;
            for ( int i = 0; i < 4; ++i )
            {
                cairo_user_to_device ( mCairoContext, &cx[i], &cy[i] );
                if ( cx[i] < dx1 )
                {
                    dx1 = cx[i];
                }
                if ( cy[i] < dy1 )
                {
                    dy1 = cy[i];
                }
                if ( cx[i] > dx2 )
                {
                    dx2 = cx[i];
                }
                if ( cy[i] > dy2 )
                {
                    dy2 = cy[i];
                }
            }
            mPickBounds[mPickId] = { dx1, dy1, dx2, dy2 };
        }
        cairo_new_path ( mCairoContext );
        // Fill path on pick surface for hit testing
        if ( mPickId > 0 && mPickContext )
        {
            cairo_append_path ( mPickContext, path.GetCairoPath() );
            cairo_set_source_rgba ( mPickContext, 0, 0, 0, mPickId / 255.0 );
            cairo_fill ( mPickContext );
        }
    }

    void CairoCanvas::DrawImage ( const uint8_t* aPixels,
                                  size_t aImageWidth,
                                  size_t aImageHeight,
                                  size_t aImageStride,
                                  double aX,
                                  double aY,
                                  double aWidth,
                                  double aHeight,
                                  double aOpacity )
    {
        if ( mCairoContext == nullptr || aPixels == nullptr || aImageWidth == 0 || aImageHeight == 0 || aWidth <= 0.0 || aHeight <= 0.0 )
        {
            return;
        }

        const double opacity = std::clamp ( aOpacity, 0.0, 1.0 );
        if ( opacity <= 0.0 )
        {
            return;
        }

        const int cairoStride = cairo_format_stride_for_width ( CAIRO_FORMAT_ARGB32, static_cast<int> ( aImageWidth ) );
        std::vector<uint8_t> cairoPixels ( static_cast<size_t> ( cairoStride ) * aImageHeight, 0u );
        for ( size_t y = 0; y < aImageHeight; ++y )
        {
            const uint8_t* sourceRow = aPixels + ( y * aImageStride );
            uint8_t* destRow = cairoPixels.data() + ( y * static_cast<size_t> ( cairoStride ) );
            for ( size_t x = 0; x < aImageWidth; ++x )
            {
                const uint8_t r = sourceRow[x * 4 + 0];
                const uint8_t g = sourceRow[x * 4 + 1];
                const uint8_t b = sourceRow[x * 4 + 2];
                const uint8_t a = sourceRow[x * 4 + 3];
                const uint8_t outAlpha = static_cast<uint8_t> ( std::round ( static_cast<double> ( a ) * opacity ) );
                const uint8_t outRed = static_cast<uint8_t> ( ( static_cast<uint16_t> ( r ) * outAlpha + 127u ) / 255u );
                const uint8_t outGreen = static_cast<uint8_t> ( ( static_cast<uint16_t> ( g ) * outAlpha + 127u ) / 255u );
                const uint8_t outBlue = static_cast<uint8_t> ( ( static_cast<uint16_t> ( b ) * outAlpha + 127u ) / 255u );

                // Cairo ARGB32 on little-endian stores bytes as BGRA.
                destRow[x * 4 + 0] = outBlue;
                destRow[x * 4 + 1] = outGreen;
                destRow[x * 4 + 2] = outRed;
                destRow[x * 4 + 3] = outAlpha;
            }
        }

        cairo_surface_t* imageSurface = cairo_image_surface_create_for_data ( cairoPixels.data(),
                                        CAIRO_FORMAT_ARGB32,
                                        static_cast<int> ( aImageWidth ),
                                        static_cast<int> ( aImageHeight ),
                                        cairoStride );
        if ( cairo_surface_status ( imageSurface ) != CAIRO_STATUS_SUCCESS )
        {
            cairo_surface_destroy ( imageSurface );
            return;
        }

        cairo_save ( mCairoContext );
        cairo_translate ( mCairoContext, aX, aY );
        cairo_scale ( mCairoContext,
                      aWidth / static_cast<double> ( aImageWidth ),
                      aHeight / static_cast<double> ( aImageHeight ) );
        cairo_set_source_surface ( mCairoContext, imageSurface, 0.0, 0.0 );
        cairo_pattern_set_filter ( cairo_get_source ( mCairoContext ), CAIRO_FILTER_BILINEAR );
        cairo_paint ( mCairoContext );
        cairo_restore ( mCairoContext );

        cairo_surface_destroy ( imageSurface );
    }

    PangoTextLayout& CairoCanvas::GetTextLayoutCache ( const std::string& aFontFamily,
            double aFontSize,
            int aFontWeight,
            int aFontStyle ) const
    {
        if ( !mTextCache )
        {
            // When the global FontDatabase has been initialised the
            // default PangoTextLayout constructor produces a context
            // bound to the (cairo-compatible) shared font map.  When it
            // has not, the default constructor would fall back to a
            // pangoft2 context, which is incompatible with cairo
            // rendering and crashes pango_cairo_show_layout.  Bind to
            // the canvas's own cairo context in that case.
            if ( FontDatabase::GetFontMap() == nullptr && mCairoContext != nullptr )
            {
                PangoContext* ctx = pango_cairo_create_context ( mCairoContext );
                mTextCache = std::make_unique<PangoTextLayout> ( ctx );
            }
            else
            {
                mTextCache = std::make_unique<PangoTextLayout>();
            }
        }
        // Setters skip work when the value is unchanged, so calling
        // them every frame is cheap once the layout has stabilised.
        mTextCache->SetFontFamily ( aFontFamily );
        mTextCache->SetFontSize ( aFontSize );
        mTextCache->SetFontWeight ( aFontWeight );
        mTextCache->SetFontStyle ( aFontStyle );
        // Make sure wrap is disabled for inline draw paths.
        mTextCache->SetWrapWidth ( -1.0 );
        return *mTextCache;
    }

    void CairoCanvas::DrawText ( const std::string& aText, double aX, double aY,
                                 const std::string& aFontFamily, double aFontSize,
                                 int aFontWeight, int aFontStyle )
    {
        if ( aText.empty() || mCairoContext == nullptr )
        {
            return;
        }

        PangoTextLayout& cache = GetTextLayoutCache ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        cache.SetText ( aText );
        PangoLayout* layout = cache.GetPangoLayout();

        // Move to the text position. SVG text y is the baseline, but Pango
        // draws from the top-left of the layout. We offset by ascent.
        const double baselineOffset = cache.GetBaseline();

        cairo_save ( mCairoContext );
        cairo_move_to ( mCairoContext, aX, aY - baselineOffset );

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_push_group ( mCairoContext );
        }

        // Render fill
        if ( std::holds_alternative<Color> ( mFillColor ) )
        {
            Color& fill = std::get<Color> ( mFillColor );
            cairo_set_source_rgba ( mCairoContext, fill.R(), fill.G(), fill.B(),
                                    ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity );
            pango_cairo_update_layout ( mCairoContext, layout );
            pango_cairo_show_layout ( mCairoContext, layout );
        }

        // Render stroke
        if ( std::holds_alternative<Color> ( mStrokeColor ) )
        {
            Color& stroke = std::get<Color> ( mStrokeColor );
            cairo_set_line_width ( mCairoContext, mStrokeWidth );
            cairo_set_source_rgba ( mCairoContext, stroke.R(), stroke.G(), stroke.B(),
                                    ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity );
            cairo_move_to ( mCairoContext, aX, aY - baselineOffset );
            pango_cairo_layout_path ( mCairoContext, layout );
            cairo_stroke ( mCairoContext );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_pop_group_to_source ( mCairoContext );
            cairo_paint_with_alpha ( mCairoContext, mOpacity );
        }

        cairo_restore ( mCairoContext );
    }

    double CairoCanvas::MeasureText ( const std::string& aText,
                                      const std::string& aFontFamily, double aFontSize,
                                      int aFontWeight, int aFontStyle ) const
    {
        if ( aText.empty() || mCairoContext == nullptr )
        {
            return 0.0;
        }

        PangoTextLayout& cache = GetTextLayoutCache ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        cache.SetText ( aText );
        return cache.GetTextWidth();
    }

    void CairoCanvas::DrawTextOnPath ( const std::string& aText,
                                       const Path& aPath,
                                       double aStartOffset,
                                       const std::string& aFontFamily, double aFontSize,
                                       int aFontWeight, int aFontStyle,
                                       bool aReverse, bool aClosed )
    {
        if ( aText.empty() || mCairoContext == nullptr )
        {
            return;
        }

        PangoTextLayout& cache = GetTextLayoutCache ( aFontFamily, aFontSize, aFontWeight, aFontStyle );
        cache.SetText ( aText );
        PangoLayout* layout = cache.GetPangoLayout();

        // Get the baseline offset for vertical centering.
        const double baselineOffset = cache.GetBaseline();

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_push_group ( mCairoContext );
        }

        // SVG2 §11.5 step 8.5.1.8: precompute path length for hidden-flag check.
        double pathLength = aPath.GetTotalLength();

        // Render character-by-character along the path.
        double distance = aStartOffset;
        size_t pos = 0;
        while ( pos < aText.size() )
        {
            // Determine the length of the current UTF-8 character.
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

            // Measure this character's advance width without copying
            // the substring: pango_layout_set_text takes a length
            // argument, so we can point straight into aText's buffer.
            pango_layout_set_text ( layout, aText.data() + pos, charLen );
            int charWidth = 0, charHeight = 0;
            pango_layout_get_pixel_size ( layout, &charWidth, &charHeight );
            double advance = static_cast<double> ( charWidth );

            // SVG2 §11.5 step 8.5.1: compute midpoint along path.
            double mid = distance + advance * 0.5;

            // For side="right", reverse the direction along the path.
            double queryMid = aReverse ? ( pathLength - mid ) : mid;

            // Closed path text wrapping: wrap around instead of hiding.
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
                // SVG2 §11.5 step 8.5.1.8: hide characters whose midpoint is off the path.
                if ( queryMid < 0.0 || queryMid > pathLength )
                {
                    distance += advance;
                    pos += static_cast<size_t> ( charLen );
                    continue;
                }
            }

            PathPoint pt = aPath.GetPointAtLength ( queryMid );
            double angle = aReverse ? ( pt.angle + M_PI ) : pt.angle;

            cairo_save ( mCairoContext );
            cairo_translate ( mCairoContext, pt.x, pt.y );
            cairo_rotate ( mCairoContext, angle );
            // Center the glyph horizontally and baseline-align vertically.
            cairo_translate ( mCairoContext, -advance * 0.5, -baselineOffset );

            // Render fill.
            if ( std::holds_alternative<Color> ( mFillColor ) )
            {
                Color& fill = std::get<Color> ( mFillColor );
                cairo_set_source_rgba ( mCairoContext, fill.R(), fill.G(), fill.B(),
                                        ( mFillOpacity >= 1.0 ) ? fill.A() : mFillOpacity );
                cairo_move_to ( mCairoContext, 0, 0 );
                pango_cairo_update_layout ( mCairoContext, layout );
                pango_cairo_show_layout ( mCairoContext, layout );
            }

            // Render stroke.
            if ( std::holds_alternative<Color> ( mStrokeColor ) )
            {
                Color& stroke = std::get<Color> ( mStrokeColor );
                cairo_set_line_width ( mCairoContext, mStrokeWidth );
                cairo_set_source_rgba ( mCairoContext, stroke.R(), stroke.G(), stroke.B(),
                                        ( mStrokeOpacity >= 1.0 ) ? stroke.A() : mStrokeOpacity );
                cairo_move_to ( mCairoContext, 0, 0 );
                pango_cairo_layout_path ( mCairoContext, layout );
                cairo_stroke ( mCairoContext );
            }

            cairo_restore ( mCairoContext );

            distance += advance;
            pos += static_cast<size_t> ( charLen );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_pop_group_to_source ( mCairoContext );
            cairo_paint_with_alpha ( mCairoContext, mOpacity );
        }

        // We bypassed PangoTextLayout::SetText for the per-character
        // measurements above, so its cached last-text no longer
        // reflects the layout's actual text. Invalidate so the next
        // SetText call re-installs the text on the layout.
        cache.InvalidateTextCache();
    }

    void CairoCanvas::SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio )
    {
        // Follows https://www.w3.org/TR/SVG2/coords.html#ComputingAViewportsTransform
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

        PreserveAspectRatio::MinMidMax x_align{aPreserveAspectRatio.GetAlignX() };
        PreserveAspectRatio::MinMidMax y_align{aPreserveAspectRatio.GetAlignY() };

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

        cairo_matrix_t transform
        {
            scale_x,     0.0,
            0.0,         scale_y,
            translate_x, translate_y
        };
        cairo_set_matrix ( mCairoContext, &transform );
        if ( mPickContext )
        {
            cairo_set_matrix ( mPickContext, &transform );
        }
    }
    void CairoCanvas::SetTransform ( const Matrix2x3& aMatrix )
    {
        cairo_matrix_t transform
        {
            aMatrix[0], aMatrix[1],
            aMatrix[2], aMatrix[3],
            aMatrix[4], aMatrix[5]
        };
        cairo_set_matrix ( mCairoContext, &transform );
        if ( mPickContext )
        {
            cairo_set_matrix ( mPickContext, &transform );
        }
    }
    void CairoCanvas::Transform ( const Matrix2x3& aMatrix )
    {
        cairo_matrix_t transform
        {
            aMatrix[0], aMatrix[1],
            aMatrix[2], aMatrix[3],
            aMatrix[4], aMatrix[5]
        };
        cairo_transform ( mCairoContext, &transform );
        if ( mPickContext )
        {
            cairo_transform ( mPickContext, &transform );
        }
    }
    void CairoCanvas::Save()
    {
        cairo_save ( mCairoContext );
        if ( mPickContext )
        {
            cairo_save ( mPickContext );
        }
    }
    void CairoCanvas::Restore()
    {
        cairo_restore ( mCairoContext );
        if ( mPickContext )
        {
            cairo_restore ( mPickContext );
        }
    }
    void* CairoCanvas::GetNativeSurface() const
    {
        return mCairoSurface;
    }

    void CairoCanvas::PushGroup()
    {
        cairo_push_group ( mCairoContext );
    }

    void CairoCanvas::PopGroup()
    {
        cairo_pop_group_to_source ( mCairoContext );
        cairo_paint ( mCairoContext );
    }

    // IIR exponential blur — in-place, zero allocations, integer fixed-point.
    // Forward+backward pass per axis approximates Gaussian blur.
    // O(1) per pixel regardless of sigma, vs O(radius) for box blur.

    // Convert Gaussian sigma to IIR alpha parameter (0..256 fixed-point).
    // For the bidirectional exponential filter, the exact relationship is:
    //   sigma^2 = 2(1-alpha)/alpha^2
    //   alpha   = (sqrt(2*sigma^2 + 1) - 1) / sigma^2
    static int SigmaToAlpha256 ( double aSigma )
    {
        if ( aSigma <= 0.0 )
        {
            return 256;    // alpha=1 means no smoothing
        }
        double s2 = aSigma * aSigma;
        double alpha = ( std::sqrt ( 2.0 * s2 + 1.0 ) - 1.0 ) / s2;
        return std::clamp ( static_cast<int> ( alpha * 256.0 + 0.5 ), 1, 255 );
    }

    static void ExpBlurH ( uint8_t* aData, int aWidth, int aHeight, int aStride, int aAlpha )
    {
        for ( int y = 0; y < aHeight; y++ )
        {
            uint8_t* row = aData + y * aStride;
            // Forward pass (left to right)
            int zB = row[0], zG = row[1], zR = row[2], zA = row[3];
            for ( int x = 1; x < aWidth; x++ )
            {
                uint8_t* p = row + x * 4;
                zB += ( ( static_cast<int> ( p[0] ) - zB ) * aAlpha ) >> 8;
                zG += ( ( static_cast<int> ( p[1] ) - zG ) * aAlpha ) >> 8;
                zR += ( ( static_cast<int> ( p[2] ) - zR ) * aAlpha ) >> 8;
                zA += ( ( static_cast<int> ( p[3] ) - zA ) * aAlpha ) >> 8;
                p[0] = static_cast<uint8_t> ( zB );
                p[1] = static_cast<uint8_t> ( zG );
                p[2] = static_cast<uint8_t> ( zR );
                p[3] = static_cast<uint8_t> ( zA );
            }
            // Backward pass (right to left)
            uint8_t* last = row + ( aWidth - 1 ) * 4;
            zB = last[0];
            zG = last[1];
            zR = last[2];
            zA = last[3];
            for ( int x = aWidth - 2; x >= 0; x-- )
            {
                uint8_t* p = row + x * 4;
                zB += ( ( static_cast<int> ( p[0] ) - zB ) * aAlpha ) >> 8;
                zG += ( ( static_cast<int> ( p[1] ) - zG ) * aAlpha ) >> 8;
                zR += ( ( static_cast<int> ( p[2] ) - zR ) * aAlpha ) >> 8;
                zA += ( ( static_cast<int> ( p[3] ) - zA ) * aAlpha ) >> 8;
                p[0] = static_cast<uint8_t> ( zB );
                p[1] = static_cast<uint8_t> ( zG );
                p[2] = static_cast<uint8_t> ( zR );
                p[3] = static_cast<uint8_t> ( zA );
            }
        }
    }

    static void ExpBlurV ( uint8_t* aData, int aWidth, int aHeight, int aStride, int aAlpha )
    {
        for ( int x = 0; x < aWidth; x++ )
        {
            uint8_t* col = aData + x * 4;
            // Forward pass (top to bottom)
            int zB = col[0], zG = col[1], zR = col[2], zA = col[3];
            for ( int y = 1; y < aHeight; y++ )
            {
                uint8_t* p = col + y * aStride;
                zB += ( ( static_cast<int> ( p[0] ) - zB ) * aAlpha ) >> 8;
                zG += ( ( static_cast<int> ( p[1] ) - zG ) * aAlpha ) >> 8;
                zR += ( ( static_cast<int> ( p[2] ) - zR ) * aAlpha ) >> 8;
                zA += ( ( static_cast<int> ( p[3] ) - zA ) * aAlpha ) >> 8;
                p[0] = static_cast<uint8_t> ( zB );
                p[1] = static_cast<uint8_t> ( zG );
                p[2] = static_cast<uint8_t> ( zR );
                p[3] = static_cast<uint8_t> ( zA );
            }
            // Backward pass (bottom to top)
            uint8_t* last = col + ( aHeight - 1 ) * aStride;
            zB = last[0];
            zG = last[1];
            zR = last[2];
            zA = last[3];
            for ( int y = aHeight - 2; y >= 0; y-- )
            {
                uint8_t* p = col + y * aStride;
                zB += ( ( static_cast<int> ( p[0] ) - zB ) * aAlpha ) >> 8;
                zG += ( ( static_cast<int> ( p[1] ) - zG ) * aAlpha ) >> 8;
                zR += ( ( static_cast<int> ( p[2] ) - zR ) * aAlpha ) >> 8;
                zA += ( ( static_cast<int> ( p[3] ) - zA ) * aAlpha ) >> 8;
                p[0] = static_cast<uint8_t> ( zB );
                p[1] = static_cast<uint8_t> ( zG );
                p[2] = static_cast<uint8_t> ( zR );
                p[3] = static_cast<uint8_t> ( zA );
            }
        }
    }

    void CairoCanvas::ApplyDropShadow ( double aDx, double aDy,
                                        double aStdDeviationX, double aStdDeviationY,
                                        const Color& aFloodColor, double aFloodOpacity )
    {
        // Pop the content group — cairo internally restores the state saved by push_group.
        cairo_pattern_t* contentPattern = cairo_pop_group ( mCairoContext );

        // -- Step 1: Render the shadow into its own group so we can blur it. --
        cairo_push_group ( mCairoContext );

        // Offset by (dx, dy) in user-space for the shadow position.
        cairo_save ( mCairoContext );
        cairo_translate ( mCairoContext, aDx, aDy );

        // Set flood color as the source; use content alpha as the mask.
        double opacity = std::clamp ( aFloodOpacity, 0.0, 1.0 );
        cairo_set_source_rgba ( mCairoContext,
                                aFloodColor.R(), aFloodColor.G(), aFloodColor.B(),
                                opacity );
        cairo_mask ( mCairoContext, contentPattern );
        cairo_restore ( mCairoContext );

        // Pop the shadow group — we now have a shadow pattern to blur.
        cairo_pattern_t* shadowPattern = cairo_pop_group ( mCairoContext );

        // -- Step 2: Blur the shadow surface using IIR exponential blur. --
        int alphaX = SigmaToAlpha256 ( aStdDeviationX );
        int alphaY = SigmaToAlpha256 ( aStdDeviationY );
        if ( alphaX < 256 || alphaY < 256 )
        {
            cairo_surface_t* shadowSurface = nullptr;
            if ( cairo_pattern_get_surface ( shadowPattern, &shadowSurface ) == CAIRO_STATUS_SUCCESS
                 && shadowSurface )
            {
                cairo_surface_flush ( shadowSurface );
                int w = cairo_image_surface_get_width ( shadowSurface );
                int h = cairo_image_surface_get_height ( shadowSurface );
                if ( w > 0 && h > 0 )
                {
                    uint8_t* data = cairo_image_surface_get_data ( shadowSurface );
                    int stride = cairo_image_surface_get_stride ( shadowSurface );
                    if ( alphaX < 256 )
                    {
                        ExpBlurH ( data, w, h, stride, alphaX );
                    }
                    if ( alphaY < 256 )
                    {
                        ExpBlurV ( data, w, h, stride, alphaY );
                    }
                    cairo_surface_mark_dirty ( shadowSurface );
                }
            }
        }

        // -- Step 3: Paint the blurred shadow, then the original content on top. --
        cairo_set_source ( mCairoContext, shadowPattern );
        cairo_paint ( mCairoContext );
        cairo_pattern_destroy ( shadowPattern );

        cairo_set_source ( mCairoContext, contentPattern );
        cairo_paint ( mCairoContext );
        cairo_pattern_destroy ( contentPattern );
    }

    void CairoCanvas::InitPickSurface ( uint32_t aWidth, uint32_t aHeight )
    {
        if ( aWidth == 0 || aHeight == 0 )
        {
            return;
        }
        mPickSurface = cairo_image_surface_create ( CAIRO_FORMAT_A8, aWidth, aHeight );
        mPickContext = cairo_create ( mPickSurface );
        cairo_set_antialias ( mPickContext, CAIRO_ANTIALIAS_NONE );
        cairo_set_operator ( mPickContext, CAIRO_OPERATOR_SOURCE );
    }

    void CairoCanvas::DestroyPickSurface()
    {
        if ( mPickContext )
        {
            cairo_destroy ( mPickContext );
            mPickContext = nullptr;
        }
        if ( mPickSurface )
        {
            cairo_surface_destroy ( mPickSurface );
            mPickSurface = nullptr;
        }
    }

    void CairoCanvas::ResetPick()
    {
        mPickId = 0;
        if ( mPickContext )
        {
            cairo_save ( mPickContext );
            cairo_set_operator ( mPickContext, CAIRO_OPERATOR_CLEAR );
            cairo_paint ( mPickContext );
            cairo_restore ( mPickContext );
            cairo_set_operator ( mPickContext, CAIRO_OPERATOR_SOURCE );
        }
    }

    uint8_t CairoCanvas::PickAtPoint ( double aX, double aY ) const
    {
        if ( !mPickSurface )
        {
            return 0;
        }
        int ix = static_cast<int> ( aX );
        int iy = static_cast<int> ( aY );
        int w = cairo_image_surface_get_width ( mPickSurface );
        int h = cairo_image_surface_get_height ( mPickSurface );
        if ( ix < 0 || iy < 0 || ix >= w || iy >= h )
        {
            return 0;
        }
        cairo_surface_flush ( mPickSurface );
        const uint8_t* data = cairo_image_surface_get_data ( mPickSurface );
        int stride = cairo_image_surface_get_stride ( mPickSurface );
        return data[iy * stride + ix];
    }

    void CairoCanvas::SetClipRect ( double aX, double aY, double aWidth, double aHeight )
    {
        // Set clip in device (pixel) coordinates on the render context.
        cairo_matrix_t saved;
        cairo_get_matrix ( mCairoContext, &saved );
        cairo_identity_matrix ( mCairoContext );
        cairo_rectangle ( mCairoContext, aX, aY, aWidth, aHeight );
        cairo_clip ( mCairoContext );
        cairo_set_matrix ( mCairoContext, &saved );

        // Mirror the clip on the pick context.
        if ( mPickContext )
        {
            cairo_get_matrix ( mPickContext, &saved );
            cairo_identity_matrix ( mPickContext );
            cairo_rectangle ( mPickContext, aX, aY, aWidth, aHeight );
            cairo_clip ( mPickContext );
            cairo_set_matrix ( mPickContext, &saved );
        }
    }

    std::unique_ptr<Path> CairoCanvas::CreatePath() const
    {
        return std::make_unique<CairoPath>();
    }
}
