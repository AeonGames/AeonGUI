/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#include <iostream>
#include "aeongui/CairoCanvas.h"

namespace AeonGUI
{
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
    void CairoCanvas::Draw ( const std::vector<DrawType>& aCommands )
    {
        cairo_save ( mCairoContext );
        cairo_set_line_width ( mCairoContext, 1 );
        cairo_set_source_rgb ( mCairoContext, 0, 0, 0 );
        for ( auto i = aCommands.begin(); i != aCommands.end(); )
        {
            switch ( std::get<uint64_t> ( * ( i++ ) ) )
            {
            case 'M':
            {
                cairo_move_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                i += 2;
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_line_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                    i += 2;
                }
            }
            break;
            case 'm':
            {
                cairo_rel_move_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                i += 2;
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_rel_line_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                    i += 2;
                }
            }
            break;
            case 'Z':
            case 'z':
                cairo_close_path ( mCairoContext );
                break;
            case 'L':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_line_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                    i += 2;
                }
                break;
            case 'l':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_rel_line_to ( mCairoContext, std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) );
                    i += 2;
                }
                break;
            case 'H':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    double x;
                    double y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    cairo_line_to ( mCairoContext, std::get<double> ( *i ), y );
                    ++i;
                }
                break;
            case 'h':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_rel_line_to ( mCairoContext, std::get<double> ( *i ), 0 );
                    ++i;
                }
                break;
            case 'V':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    double x;
                    double y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    cairo_line_to ( mCairoContext, x, std::get<double> ( *i ) );
                    ++i;
                }
                break;
            case 'v':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_rel_line_to ( mCairoContext, 0, std::get<double> ( *i ) );
                    ++i;
                }
                break;
            case 'C':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_curve_to ( mCairoContext,
                                     std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                     std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ),
                                     std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) );
                    i += 6;
                }
                break;
            case 'c':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    cairo_rel_curve_to ( mCairoContext,
                                         std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                         std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ),
                                         std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) );
                    i += 6;
                }
                break;
            case 'S':
                break;
            case 's':
                break;
            case 'Q':
                break;
            case 'q':
                break;
            case 'T':
                break;
            case 't':
                break;
            case 'A':
                break;
            case 'a':
                break;
            }
        }
        cairo_stroke ( mCairoContext );
        cairo_restore ( mCairoContext );
    }
}
