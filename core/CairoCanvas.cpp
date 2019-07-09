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
#include <algorithm>
#include <cmath>
#include <limits>
#include "aeongui/CairoCanvas.h"

namespace AeonGUI
{
    /**@note the following two functions have been adapted from librsvg code. */
    static void
    path_arc_segment ( cairo_t* context,
                       bool relative,
                       double xc, double yc,
                       double th0, double th1, double rx, double ry,
                       double x_axis_rotation )
    {
        double x1, y1, x2, y2, x3, y3;
        double t;
        double th_half;
        double f, sinf, cosf;

        f = x_axis_rotation * M_PI / 180.0;
        sinf = std::sin ( f );
        cosf = std::cos ( f );

        th_half = 0.5 * ( th1 - th0 );
        t = ( 8.0 / 3.0 ) * std::sin ( th_half * 0.5 ) * std::sin ( th_half * 0.5 ) / std::sin ( th_half );
        x1 = rx * ( std::cos ( th0 ) - t * std::sin ( th0 ) );
        y1 = ry * ( std::sin ( th0 ) + t * std::cos ( th0 ) );
        x3 = rx * std::cos ( th1 );
        y3 = ry * std::sin ( th1 );
        x2 = x3 + rx * ( t * std::sin ( th1 ) );
        y2 = y3 + ry * ( -t * std::cos ( th1 ) );
        if ( !relative )
        {
            cairo_curve_to (   context,
                               xc + cosf * x1 - sinf * y1,
                               yc + sinf * x1 + cosf * y1,
                               xc + cosf * x2 - sinf * y2,
                               yc + sinf * x2 + cosf * y2,
                               xc + cosf * x3 - sinf * y3,
                               yc + sinf * x3 + cosf * y3 );
        }
        else
        {
            cairo_rel_curve_to (   context,
                                   xc + cosf * x1 - sinf * y1,
                                   yc + sinf * x1 + cosf * y1,
                                   xc + cosf * x2 - sinf * y2,
                                   yc + sinf * x2 + cosf * y2,
                                   xc + cosf * x3 - sinf * y3,
                                   yc + sinf * x3 + cosf * y3 );
        }
    }

    static void
    path_arc (   cairo_t* context,
                 bool relative,
                 double x1, double y1,
                 double rx, double ry,
                 double x_axis_rotation,
                 bool large_arc_flag, bool sweep_flag,
                 double x2, double y2 )
    {

        /* See Appendix F.6 Elliptical arc implementation notes
           http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes */

        double f, sinf, cosf;
        double x1_, y1_;
        double cx_, cy_, cx, cy;
        double gamma;
        double theta1, delta_theta;
        double k1, k2, k3, k4, k5;

        int i, n_segs;

        if ( x1 == x2 && y1 == y2 )
        {
            return;
        }

        /* X-axis */
        f = x_axis_rotation * M_PI / 180.0;
        sinf = std::sin ( f );
        cosf = std::cos ( f );

        rx = std::abs ( rx );
        ry = std::abs ( ry );

        if ( ( rx < std::numeric_limits<double>::epsilon() ) || ( ry < std::numeric_limits<double>::epsilon() ) )
        {
            cairo_line_to ( context, x2, y2 );
            return;
        }

        k1 = ( x1 - x2 ) / 2;
        k2 = ( y1 - y2 ) / 2;

        x1_ = cosf * k1 + sinf * k2;
        y1_ = -sinf * k1 + cosf * k2;

        gamma = ( x1_ * x1_ ) / ( rx * rx ) + ( y1_ * y1_ ) / ( ry * ry );
        if ( gamma > 1 )
        {
            rx *= std::sqrt ( gamma );
            ry *= std::sqrt ( gamma );
        }

        /* Compute the center */

        k1 = rx * rx * y1_ * y1_ + ry * ry * x1_ * x1_;
        if ( k1 == 0 )
        {
            return;
        }

        k1 = std::sqrt ( std::abs ( ( rx * rx * ry * ry ) / k1 - 1 ) );
        if ( sweep_flag == large_arc_flag )
        {
            k1 = -k1;
        }

        cx_ = k1 * rx * y1_ / ry;
        cy_ = -k1 * ry * x1_ / rx;

        cx = cosf * cx_ - sinf * cy_ + ( x1 + x2 ) / 2;
        cy = sinf * cx_ + cosf * cy_ + ( y1 + y2 ) / 2;

        /* Compute start angle */

        k1 = ( x1_ - cx_ ) / rx;
        k2 = ( y1_ - cy_ ) / ry;
        k3 = ( -x1_ - cx_ ) / rx;
        k4 = ( -y1_ - cy_ ) / ry;

        k5 = std::sqrt ( std::abs ( k1 * k1 + k2 * k2 ) );
        if ( k5 == 0 )
        {
            return;
        }

        k5 = k1 / k5;
        k5 = std::clamp ( k5, -1.0, 1.0 );
        theta1 = std::acos ( k5 );
        if ( k2 < 0 )
        {
            theta1 = -theta1;
        }

        /* Compute delta_theta */

        k5 = std::sqrt ( std::abs ( ( k1 * k1 + k2 * k2 ) * ( k3 * k3 + k4 * k4 ) ) );
        if ( k5 == 0 )
        {
            return;
        }

        k5 = ( k1 * k3 + k2 * k4 ) / k5;
        k5 = std::clamp ( k5, -1.0, 1.0 );
        delta_theta = std::acos ( k5 );
        if ( k1 * k4 - k3 * k2 < 0 )
        {
            delta_theta = -delta_theta;
        }

        if ( sweep_flag && delta_theta < 0 )
        {
            delta_theta += M_PI * 2;
        }
        else if ( !sweep_flag && delta_theta > 0 )
        {
            delta_theta -= M_PI * 2;
        }

        /* Now draw the arc */

        n_segs = std::ceil ( std::abs ( delta_theta / ( M_PI * 0.5 + 0.001 ) ) );

        for ( i = 0; i < n_segs; i++ )
        {
            path_arc_segment ( context, relative,
                               cx, cy,
                               theta1 + i * delta_theta / n_segs,
                               theta1 + ( i + 1 ) * delta_theta / n_segs,
                               rx, ry, x_axis_rotation );
        }
    }

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
        cairo_set_line_width ( mCairoContext, 1 );
        cairo_set_source_rgb ( mCairoContext, 0, 0, 0 );
        uint64_t last_cmd{};
        Vector2 last_c_ctrl{};
        Vector2 last_q_ctrl{};
        for ( auto i = aCommands.begin(); i != aCommands.end(); )
        {
            uint64_t cmd{std::get<uint64_t> ( * ( i ) ) };
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
                    last_c_ctrl = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    cairo_curve_to ( mCairoContext,
                                     std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                     last_c_ctrl[0], last_c_ctrl[1],
                                     std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) );
                    i += 6;
                }
                break;
            case 'c':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    last_c_ctrl = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    cairo_rel_curve_to ( mCairoContext,
                                         std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                         last_c_ctrl[0], last_c_ctrl[1],
                                         std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) );
                    i += 6;
                }
                break;
            case 'S':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    double x, y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    last_c_ctrl = ( last_cmd == 'C' || last_cmd == 'S' ) ? Vector2{ ( 2 * x ) - last_c_ctrl[0], ( 2 * y ) - last_c_ctrl[1]}:
                                  Vector2{};
                    cairo_curve_to ( mCairoContext,
                                     last_c_ctrl[0], last_c_ctrl[1],
                                     std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                     std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) );
                    last_c_ctrl = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    i += 4;
                }
                break;
            case 's':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    last_c_ctrl = ( last_cmd == 'c' || last_cmd == 's' ) ? Vector2{-last_c_ctrl[0], -last_c_ctrl[1]}:
                                  Vector2{};
                    cairo_rel_curve_to ( mCairoContext,
                                         last_c_ctrl[0], last_c_ctrl[1],
                                         std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ),
                                         std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) );
                    last_c_ctrl = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    i += 4;
                }
                break;
            case 'Q':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    double x, y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    last_q_ctrl = {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 P2{std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    Vector2 Q1{Vector2{x* ( 1.0 / 3.0 ), y* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + P2* ( 1.0 / 3.0 ) };
                    cairo_curve_to ( mCairoContext,
                                     Q1[0], Q1[1],
                                     Q2[0], Q2[1],
                                     P2[0], P2[1] );
                    i += 4;
                }
                break;
            case 'q':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 P2{std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    Vector2 Q1{last_q_ctrl* ( 2.0 / 3.0 ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + P2* ( 1.0 / 3.0 ) };
                    cairo_rel_curve_to ( mCairoContext,
                                         Q1[0], Q1[1],
                                         Q2[0], Q2[1],
                                         P2[0], P2[1] );
                    i += 4;
                }
                break;
            case 'T':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    double x, y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    last_q_ctrl = ( last_cmd == 'Q' || last_cmd == 'T' ) ? Vector2{ ( 2 * x ) - last_q_ctrl[0], ( 2 * y ) - last_q_ctrl[1]}:
                                  Vector2{};
                    Vector2 P2{std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q1{Vector2{x* ( 1.0 / 3.0 ), y* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + P2* ( 1.0 / 3.0 ) };
                    cairo_curve_to ( mCairoContext,
                                     Q1[0], Q1[1],
                                     Q2[0], Q2[1],
                                     P2[0], P2[1] );
                    i += 2;
                }
                break;
            case 't':
                while ( i != aCommands.end() && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = ( last_cmd == 'q' || last_cmd == 't' ) ? Vector2{-last_q_ctrl[0], -last_q_ctrl[1]}:
                                  Vector2{};
                    Vector2 P2{std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q1{last_q_ctrl* ( 2.0 / 3.0 ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + P2* ( 1.0 / 3.0 ) };
                    cairo_curve_to ( mCairoContext,
                                     Q1[0], Q1[1],
                                     Q2[0], Q2[1],
                                     P2[0], P2[1] );
                    i += 2;
                }
                break;
            case 'A':
                while ( i != aCommands.end() && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    double x, y;
                    cairo_get_current_point ( mCairoContext, &x, &y );
                    path_arc ( mCairoContext, false,
                               x, y,
                               std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ),
                               std::get<double> ( * ( i + 2 ) ),
                               std::get<bool> ( * ( i + 3 ) ), std::get<bool> ( * ( i + 4 ) ),
                               std::get<double> ( * ( i + 5 ) ), std::get<double> ( * ( i + 6 ) )
                             );
                    i += 7;
                }
                break;
            case 'a':
                while ( i != aCommands.end() && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    path_arc ( mCairoContext, true,
                               0.0, 0.0,
                               std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ),
                               std::get<double> ( * ( i + 2 ) ),
                               std::get<bool> ( * ( i + 3 ) ), std::get<bool> ( * ( i + 4 ) ),
                               std::get<double> ( * ( i + 5 ) ), std::get<double> ( * ( i + 6 ) )
                             );
                    i += 7;
                }
                break;
            }
            last_cmd = cmd;
        }
        cairo_stroke ( mCairoContext );
    }
}
