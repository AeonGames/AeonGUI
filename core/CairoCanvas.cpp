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
#include <limits>
#include "aeongui/CairoCanvas.hpp"
#include "aeongui/CairoPath.hpp"
#include "aeongui/PangoTextLayout.hpp"
#include "aeongui/FontDatabase.hpp"

namespace AeonGUI
{
    CairoCanvas::CairoCanvas () = default;
    CairoCanvas::CairoCanvas ( uint32_t aWidth, uint32_t aHeight ) :
        mCairoSurface{cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight ) },
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) }
    {}

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
    }

    const uint8_t* CairoCanvas::GetPixels() const
    {
        return cairo_image_surface_get_data ( mCairoSurface );
    }

    size_t CairoCanvas::GetWidth() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_width ( mCairoSurface ) );
    }
    size_t CairoCanvas::GetHeight() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_height ( mCairoSurface ) );
    }
    size_t CairoCanvas::GetStride() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_stride ( mCairoSurface ) );
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
        cairo_new_path ( mCairoContext );
    }

    static PangoFontDescription* CreateFontDescription ( const std::string& aFontFamily, double aFontSize,
            int aFontWeight, int aFontStyle )
    {
        PangoFontDescription* desc = pango_font_description_new();
        pango_font_description_set_family ( desc, aFontFamily.c_str() );
        pango_font_description_set_size ( desc, static_cast<gint> ( aFontSize * PANGO_SCALE ) );

        // Map CSS font-weight (100-900) to Pango weight
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

        // Map CSS font-style: 0 = normal, 1 = italic, 2 = oblique
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

    void CairoCanvas::DrawText ( const std::string& aText, double aX, double aY,
                                 const std::string& aFontFamily, double aFontSize,
                                 int aFontWeight, int aFontStyle )
    {
        if ( aText.empty() || mCairoContext == nullptr )
        {
            return;
        }

        PangoContext* pangoContext = FontDatabase::GetFontMap()
                                     ? FontDatabase::CreateContext()
                                     : pango_cairo_create_context ( mCairoContext );
        PangoLayout* layout = pango_layout_new ( pangoContext );
        PangoFontDescription* desc = CreateFontDescription ( aFontFamily, aFontSize, aFontWeight, aFontStyle );

        pango_layout_set_font_description ( layout, desc );
        pango_layout_set_text ( layout, aText.c_str(), -1 );

        // Move to the text position. SVG text y is the baseline, but Pango
        // draws from the top-left of the layout. We offset by ascent.
        PangoLayoutIter* iter = pango_layout_get_iter ( layout );
        int baseline = pango_layout_iter_get_baseline ( iter );
        pango_layout_iter_free ( iter );

        cairo_save ( mCairoContext );
        cairo_move_to ( mCairoContext, aX, aY - static_cast<double> ( baseline ) / PANGO_SCALE );

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
            cairo_move_to ( mCairoContext, aX, aY - static_cast<double> ( baseline ) / PANGO_SCALE );
            pango_cairo_layout_path ( mCairoContext, layout );
            cairo_stroke ( mCairoContext );
        }

        if ( mOpacity < 1.0 && mOpacity > 0.0 )
        {
            cairo_pop_group_to_source ( mCairoContext );
            cairo_paint_with_alpha ( mCairoContext, mOpacity );
        }

        cairo_restore ( mCairoContext );

        pango_font_description_free ( desc );
        g_object_unref ( layout );
        g_object_unref ( pangoContext );
    }

    double CairoCanvas::MeasureText ( const std::string& aText,
                                      const std::string& aFontFamily, double aFontSize,
                                      int aFontWeight, int aFontStyle ) const
    {
        if ( aText.empty() || mCairoContext == nullptr )
        {
            return 0.0;
        }

        PangoContext* pangoContext = FontDatabase::GetFontMap()
                                     ? FontDatabase::CreateContext()
                                     : pango_cairo_create_context ( mCairoContext );
        PangoLayout* layout = pango_layout_new ( pangoContext );
        PangoFontDescription* desc = CreateFontDescription ( aFontFamily, aFontSize, aFontWeight, aFontStyle );

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
    }
    void* CairoCanvas::GetNativeSurface() const
    {
        return mCairoSurface;
    }
}
