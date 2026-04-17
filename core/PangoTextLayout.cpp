/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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

#include "PangoTextLayout.hpp"
#include "aeongui/FontDatabase.hpp"
#include <pango/pango.h>
#include <pango/pangocairo.h>
#include <stdexcept>
namespace AeonGUI
{
    PangoTextLayout::PangoTextLayout()
    {
        if ( FontDatabase::GetFontMap() )
        {
            mPangoContext = FontDatabase::CreateContext();
        }
        else
        {
            // Fallback: create context from the default Cairo font map.
            PangoFontMap* fontMap = pango_cairo_font_map_get_default();
            mPangoContext = pango_font_map_create_context ( fontMap );
        }
        mLayout = pango_layout_new ( mPangoContext );
        mFontDescription = pango_font_description_new();
        UpdateFontDescription();
    }

    PangoTextLayout::~PangoTextLayout()
    {
        if ( mFontDescription )
        {
            pango_font_description_free ( mFontDescription );
        }
        if ( mLayout )
        {
            g_object_unref ( mLayout );
        }
        if ( mPangoContext )
        {
            g_object_unref ( mPangoContext );
        }
    }

    void PangoTextLayout::UpdateFontDescription()
    {
        pango_font_description_set_family ( mFontDescription, mFontFamily.c_str() );
        pango_font_description_set_absolute_size ( mFontDescription, mFontSize * PANGO_SCALE );

        PangoWeight weight = PANGO_WEIGHT_NORMAL;
        if ( mFontWeight <= 100 )
        {
            weight = PANGO_WEIGHT_THIN;
        }
        else if ( mFontWeight <= 200 )
        {
            weight = PANGO_WEIGHT_ULTRALIGHT;
        }
        else if ( mFontWeight <= 300 )
        {
            weight = PANGO_WEIGHT_LIGHT;
        }
        else if ( mFontWeight <= 400 )
        {
            weight = PANGO_WEIGHT_NORMAL;
        }
        else if ( mFontWeight <= 500 )
        {
            weight = PANGO_WEIGHT_MEDIUM;
        }
        else if ( mFontWeight <= 600 )
        {
            weight = PANGO_WEIGHT_SEMIBOLD;
        }
        else if ( mFontWeight <= 700 )
        {
            weight = PANGO_WEIGHT_BOLD;
        }
        else if ( mFontWeight <= 800 )
        {
            weight = PANGO_WEIGHT_ULTRABOLD;
        }
        else
        {
            weight = PANGO_WEIGHT_HEAVY;
        }
        pango_font_description_set_weight ( mFontDescription, weight );

        PangoStyle style = PANGO_STYLE_NORMAL;
        if ( mFontStyle == 1 )
        {
            style = PANGO_STYLE_ITALIC;
        }
        else if ( mFontStyle == 2 )
        {
            style = PANGO_STYLE_OBLIQUE;
        }
        pango_font_description_set_style ( mFontDescription, style );

        pango_layout_set_font_description ( mLayout, mFontDescription );
    }

    void PangoTextLayout::SetText ( const std::string& aText )
    {
        pango_layout_set_text ( mLayout, aText.c_str(), static_cast<int> ( aText.size() ) );
    }

    void PangoTextLayout::SetFontFamily ( const std::string& aFamily )
    {
        mFontFamily = aFamily;
        UpdateFontDescription();
    }

    void PangoTextLayout::SetFontSize ( double aSize )
    {
        mFontSize = aSize;
        UpdateFontDescription();
    }

    void PangoTextLayout::SetFontWeight ( int aWeight )
    {
        mFontWeight = aWeight;
        UpdateFontDescription();
    }

    void PangoTextLayout::SetFontStyle ( int aStyle )
    {
        mFontStyle = aStyle;
        UpdateFontDescription();
    }

    double PangoTextLayout::GetTextWidth() const
    {
        int width = 0;
        int height = 0;
        pango_layout_get_pixel_size ( mLayout, &width, &height );
        return static_cast<double> ( width );
    }

    double PangoTextLayout::GetTextHeight() const
    {
        int width = 0;
        int height = 0;
        pango_layout_get_pixel_size ( mLayout, &width, &height );
        return static_cast<double> ( height );
    }

    double PangoTextLayout::GetBaseline() const
    {
        PangoLayoutIter* iter = pango_layout_get_iter ( mLayout );
        int baseline = pango_layout_iter_get_baseline ( iter );
        pango_layout_iter_free ( iter );
        return static_cast<double> ( baseline ) / PANGO_SCALE;
    }

    double PangoTextLayout::GetCharOffsetX ( int32_t aIndex ) const
    {
        PangoRectangle pos;
        pango_layout_index_to_pos ( mLayout, static_cast<int> ( aIndex ), &pos );
        return static_cast<double> ( pos.x ) / PANGO_SCALE;
    }

    PangoLayout* PangoTextLayout::GetPangoLayout() const
    {
        return mLayout;
    }
}