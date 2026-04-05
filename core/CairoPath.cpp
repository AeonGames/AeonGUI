/*
Copyright (C) 2019,2020,2025,2026 Rodrigo Jose Hernandez Cordoba

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

#include <cmath>
#include <cairo.h>
#include "CairoPath.hpp"

namespace AeonGUI
{
    static_assert ( sizeof ( cairo_path_t ) <= 32, "CairoPath: cairo_path_t size exceeds static allocation buffer" );
    /**@note the following two functions have been adapted from librsvg code. */
    static void
    path_arc_segment ( std::vector<cairo_path_data_t>& path_data,
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

        path_data.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
        path_data.emplace_back ( cairo_path_data_t{.point = {xc + cosf * x1 - sinf * y1, yc + sinf * x1 + cosf * y1}} );
        path_data.emplace_back ( cairo_path_data_t{.point = {xc + cosf * x2 - sinf * y2, yc + sinf * x2 + cosf * y2}} );
        path_data.emplace_back ( cairo_path_data_t{.point = {xc + cosf * x3 - sinf * y3, yc + sinf * x3 + cosf * y3}} );
    }

    static void
    path_arc (   std::vector<cairo_path_data_t>& path_data,
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
            path_data.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
            path_data.emplace_back ( cairo_path_data_t{.point = {x2, y2}} );
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

        size_t n_segs = static_cast<size_t> ( std::ceil ( std::abs ( delta_theta / ( M_PI * 0.5 + 0.001 ) ) ) );

        for ( size_t i = 0; i < n_segs; i++ )
        {
            path_arc_segment ( path_data,
                               cx, cy,
                               theta1 + i * delta_theta / n_segs,
                               theta1 + ( i + 1 ) * delta_theta / n_segs,
                               rx, ry, x_axis_rotation );
        }
    }

    CairoPath::CairoPath() = default;
    const cairo_path_t* CairoPath::GetCairoPath() const
    {
        return &mPath;
    }

    void CairoPath::Construct ( const std::vector<DrawType>& aCommands, size_t aPathDataHint )
    {
        Construct ( aCommands.data(), aCommands.size(), aPathDataHint );
    }

    void CairoPath::Construct ( const DrawType* aCommands, size_t aCommandCount, size_t aPathDataHint )
    {
        mPathData.clear();
        if ( aPathDataHint > 0 )
        {
            mPathData.reserve ( aPathDataHint );
        }
        else
        {
            // Upper bound: each command produces at most 3 elements (Z),
            // each argument produces at most 2 elements.
            size_t estimated = 0;
            for ( size_t idx = 0; idx < aCommandCount; ++idx )
            {
                estimated += std::holds_alternative<uint64_t> ( aCommands[idx] ) ? 3 : 2;
            }
            mPathData.reserve ( estimated );
        }
        uint64_t last_cmd{};
        Vector2 last_point{0, 0};
        Vector2 last_move{0, 0};
        Vector2 last_c_ctrl{};
        Vector2 last_q_ctrl{};
        const DrawType* end = aCommands + aCommandCount;
        for ( const DrawType * i = aCommands; i != end; )
        {
            uint64_t cmd{std::get<uint64_t> ( * ( i ) ) };
            switch ( std::get<uint64_t> ( * ( i++ ) ) )
            {
            case 'M':
            case 'm':
            {
                mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_MOVE_TO, 2}} );
                last_move = last_point = ( ( cmd == 'm' ) ? last_point : Vector2{0, 0} ) + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                i += 2;
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point += {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 2;
                }
            }
            break;
            case 'Z':
            case 'z':
                mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CLOSE_PATH, 1}} );
                mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_MOVE_TO, 2}} );
                last_point = last_move;
                mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                break;
            case 'L':
            case 'l':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point = ( ( cmd == 'l' ) ? last_point : Vector2{0, 0} ) + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 2;
                }
                break;
            case 'H':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point[0] = std::get<double> ( *i );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    ++i;
                }
                break;
            case 'h':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point[0] += std::get<double> ( *i );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    ++i;
                }
                break;
            case 'V':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point[1] = std::get<double> ( *i );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    ++i;
                }
                break;
            case 'v':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_LINE_TO, 2}} );
                    last_point[1] += std::get<double> ( *i );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    ++i;
                }
                break;
            case 'C':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_c_ctrl = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    last_point  = {std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) }} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 6;
                }
                break;
            case 'c':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0] + std::get<double> ( *i ), last_point[1] + std::get<double> ( * ( i + 1 ) ) }} );
                    last_c_ctrl = last_point + Vector2{std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    last_point += {std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 6;
                }
                break;
            case 'S':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_c_ctrl = ( last_cmd == 'C' || last_cmd == 'S' ) ? Vector2{ ( 2 * last_point[0] ) - last_c_ctrl[0], ( 2 * last_point[1] ) - last_c_ctrl[1]}:
                                  Vector2{};
                    last_point = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    last_c_ctrl = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 4;
                }
                break;
            case 's':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_c_ctrl = ( last_cmd == 'c' || last_cmd == 's' ) ? Vector2{ ( 2 * last_point[0] ) - last_c_ctrl[0], ( 2 * last_point[1] ) - last_c_ctrl[1]}:
                                  Vector2{};
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    last_c_ctrl = last_point + Vector2{std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_c_ctrl[0], last_c_ctrl[1]}} );
                    last_point += {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 4;
                }
                break;
            case 'Q':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q1{Vector2{last_point[0]* ( 1.0 / 3.0 ), last_point[1]* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    last_point = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + last_point * ( 1.0 / 3.0 ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q1[0], Q1[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q2[0], Q2[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 4;
                }
                break;
            case 'q':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = last_point + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q1{Vector2{last_point[0]* ( 1.0 / 3.0 ), last_point[1]* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    last_point += {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + last_point * ( 1.0 / 3.0 ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q1[0], Q1[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q2[0], Q2[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 4;
                }
                break;
            case 'T':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = ( last_cmd == 'Q' || last_cmd == 'T' ) ? Vector2{ ( 2 * last_point[0] ) - last_q_ctrl[0], ( 2 * last_point[1] ) - last_q_ctrl[1]}:
                                  Vector2{};
                    Vector2 Q1{Vector2{last_point[0]* ( 1.0 / 3.0 ), last_point[1]* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    last_point = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + last_point* ( 1.0 / 3.0 ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q1[0], Q1[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q2[0], Q2[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 2;
                }
                break;
            case 't':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = last_point + ( ( last_cmd == 'q' || last_cmd == 't' ) ? Vector2{ ( 2 * last_point[0] ) - last_q_ctrl[0], ( 2 * last_point[1] ) - last_q_ctrl[1]}:
                                                 Vector2{} );
                    Vector2 Q1{Vector2{last_point[0]* ( 1.0 / 3.0 ), last_point[1]* ( 1.0 / 3.0 ) } + last_q_ctrl* ( 2.0 / 3.0 ) };
                    last_point += {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    Vector2 Q2{last_q_ctrl* ( 2.0 / 3.0 ) + last_point* ( 1.0 / 3.0 ) };
                    mPathData.emplace_back ( cairo_path_data_t{{CAIRO_PATH_CURVE_TO, 4}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q1[0], Q1[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {Q2[0], Q2[1]}} );
                    mPathData.emplace_back ( cairo_path_data_t{.point = {last_point[0], last_point[1]}} );
                    i += 2;
                }
                break;
            case 'A':
                while ( i != end && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    path_arc ( mPathData,
                               last_point[0], last_point[1],
                               std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ),
                               std::get<double> ( * ( i + 2 ) ),
                               std::get<bool> ( * ( i + 3 ) ), std::get<bool> ( * ( i + 4 ) ),
                               std::get<double> ( * ( i + 5 ) ), std::get<double> ( * ( i + 6 ) )
                             );
                    last_point = {std::get<double> ( * ( i + 5 ) ), std::get<double> ( * ( i + 6 ) ) };
                    i += 7;
                }
                break;
            case 'a':
                while ( i != end && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    path_arc ( mPathData,
                               last_point[0], last_point[1],
                               std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ),
                               std::get<double> ( * ( i + 2 ) ),
                               std::get<bool> ( * ( i + 3 ) ), std::get<bool> ( * ( i + 4 ) ),
                               last_point[0] + std::get<double> ( * ( i + 5 ) ), last_point[1] + std::get<double> ( * ( i + 6 ) )
                             );
                    last_point += {std::get<double> ( * ( i + 5 ) ), std::get<double> ( * ( i + 6 ) ) };
                    i += 7;
                }
                break;
            }
            last_cmd = cmd;
        }
        mPath.status = CAIRO_STATUS_SUCCESS;
        mPath.data = mPathData.data();
        mPath.num_data = static_cast<int> ( mPathData.size() );
    }
    CairoPath::~CairoPath() = default;

    // Evaluate a cubic Bezier at parameter t.
    static void CubicBezierPoint ( double t,
                                   double x0, double y0,
                                   double x1, double y1,
                                   double x2, double y2,
                                   double x3, double y3,
                                   double& ox, double& oy )
    {
        double u = 1.0 - t;
        double uu = u * u;
        double tt = t * t;
        ox = uu * u * x0 + 3.0 * uu * t * x1 + 3.0 * u * tt * x2 + tt * t * x3;
        oy = uu * u * y0 + 3.0 * uu * t * y1 + 3.0 * u * tt * y2 + tt * t * y3;
    }

    // Evaluate the derivative of a cubic Bezier at parameter t.
    static void CubicBezierTangent ( double t,
                                     double x0, double y0,
                                     double x1, double y1,
                                     double x2, double y2,
                                     double x3, double y3,
                                     double& dx, double& dy )
    {
        double u = 1.0 - t;
        dx = 3.0 * u * u * ( x1 - x0 ) + 6.0 * u * t * ( x2 - x1 ) + 3.0 * t * t * ( x3 - x2 );
        dy = 3.0 * u * u * ( y1 - y0 ) + 6.0 * u * t * ( y2 - y1 ) + 3.0 * t * t * ( y3 - y2 );
    }

    // Approximate the arc length of a cubic Bezier using 16-point Gauss-Legendre quadrature.
    static double CubicBezierLength ( double x0, double y0,
                                      double x1, double y1,
                                      double x2, double y2,
                                      double x3, double y3 )
    {
        // 8-point Gauss-Legendre weights and abscissae on [0,1].
        static const double weights[] = {0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                                         0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
                                        };
        static const double abscissae[] = {0.0198550718, 0.1016667613, 0.2372337950, 0.4082826788,
                                           0.5917173212, 0.7627662050, 0.8983332387, 0.9801449282
                                          };
        double length = 0.0;
        for ( int j = 0; j < 8; ++j )
        {
            double dx, dy;
            CubicBezierTangent ( abscissae[j], x0, y0, x1, y1, x2, y2, x3, y3, dx, dy );
            length += weights[j] * std::sqrt ( dx * dx + dy * dy );
        }
        // Weights are for [-1,1]; multiply by 0.5 to scale to [0,1].
        return length * 0.5;
    }

    double CairoPath::GetTotalLength() const
    {
        double totalLength = 0.0;
        double curX = 0.0, curY = 0.0;
        for ( int i = 0; i < mPath.num_data; i += mPath.data[i].header.length )
        {
            cairo_path_data_t* data = &mPath.data[i];
            switch ( data->header.type )
            {
            case CAIRO_PATH_MOVE_TO:
                curX = data[1].point.x;
                curY = data[1].point.y;
                break;
            case CAIRO_PATH_LINE_TO:
            {
                double dx = data[1].point.x - curX;
                double dy = data[1].point.y - curY;
                totalLength += std::sqrt ( dx * dx + dy * dy );
                curX = data[1].point.x;
                curY = data[1].point.y;
            }
            break;
            case CAIRO_PATH_CURVE_TO:
                totalLength += CubicBezierLength ( curX, curY,
                                                   data[1].point.x, data[1].point.y,
                                                   data[2].point.x, data[2].point.y,
                                                   data[3].point.x, data[3].point.y );
                curX = data[3].point.x;
                curY = data[3].point.y;
                break;
            case CAIRO_PATH_CLOSE_PATH:
                break;
            }
        }
        return totalLength;
    }

    PathPoint CairoPath::GetPointAtLength ( double aDistance ) const
    {
        double remaining = aDistance;
        double curX = 0.0, curY = 0.0;
        double startX = 0.0, startY = 0.0;

        for ( int i = 0; i < mPath.num_data; i += mPath.data[i].header.length )
        {
            cairo_path_data_t* data = &mPath.data[i];
            switch ( data->header.type )
            {
            case CAIRO_PATH_MOVE_TO:
                curX = startX = data[1].point.x;
                curY = startY = data[1].point.y;
                break;
            case CAIRO_PATH_LINE_TO:
            {
                double dx = data[1].point.x - curX;
                double dy = data[1].point.y - curY;
                double segLen = std::sqrt ( dx * dx + dy * dy );
                if ( remaining <= segLen && segLen > 0.0 )
                {
                    double t = remaining / segLen;
                    return PathPoint{curX + dx * t, curY + dy * t, std::atan2 ( dy, dx ) };
                }
                remaining -= segLen;
                curX = data[1].point.x;
                curY = data[1].point.y;
            }
            break;
            case CAIRO_PATH_CURVE_TO:
            {
                double x0 = curX, y0 = curY;
                double x1 = data[1].point.x, y1 = data[1].point.y;
                double x2 = data[2].point.x, y2 = data[2].point.y;
                double x3 = data[3].point.x, y3 = data[3].point.y;
                double segLen = CubicBezierLength ( x0, y0, x1, y1, x2, y2, x3, y3 );
                if ( remaining <= segLen && segLen > 0.0 )
                {
                    // Binary search for the parameter t where arc length == remaining.
                    double lo = 0.0, hi = 1.0;
                    for ( int iter = 0; iter < 30; ++iter )
                    {
                        double mid = ( lo + hi ) * 0.5;
                        // Compute arc length from 0 to mid using Gauss-Legendre on [0, mid].
                        double subLen = 0.0;
                        {
                            static const double w[] = {0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                                                       0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
                                                      };
                            static const double a[] = {0.0198550718, 0.1016667613, 0.2372337950, 0.4082826788,
                                                       0.5917173212, 0.7627662050, 0.8983332387, 0.9801449282
                                                      };
                            for ( int j = 0; j < 8; ++j )
                            {
                                double tt = a[j] * mid;
                                double ddx, ddy;
                                CubicBezierTangent ( tt, x0, y0, x1, y1, x2, y2, x3, y3, ddx, ddy );
                                subLen += w[j] * std::sqrt ( ddx * ddx + ddy * ddy );
                            }
                            // Weights are for [-1,1]; multiply by 0.5 to scale to [0,1].
                            subLen *= mid * 0.5;
                        }
                        if ( subLen < remaining )
                        {
                            lo = mid;
                        }
                        else
                        {
                            hi = mid;
                        }
                    }
                    double t = ( lo + hi ) * 0.5;
                    double ox, oy, ddx, ddy;
                    CubicBezierPoint ( t, x0, y0, x1, y1, x2, y2, x3, y3, ox, oy );
                    CubicBezierTangent ( t, x0, y0, x1, y1, x2, y2, x3, y3, ddx, ddy );
                    return PathPoint{ox, oy, std::atan2 ( ddy, ddx ) };
                }
                remaining -= segLen;
                curX = x3;
                curY = y3;
            }
            break;
            case CAIRO_PATH_CLOSE_PATH:
                break;
            }
        }
        // Past the end — return the last point with the last tangent.
        return PathPoint{curX, curY, 0.0};
    }

    bool CairoPath::IsClosed() const
    {
        for ( int i = 0; i < mPath.num_data; i += mPath.data[i].header.length )
        {
            if ( mPath.data[i].header.type == CAIRO_PATH_CLOSE_PATH )
            {
                return true;
            }
        }
        return false;
    }
}
