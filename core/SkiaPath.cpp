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

#include <cmath>
#include "aeongui/SkiaPath.hpp"
#include <include/core/SkPathMeasure.h>

namespace AeonGUI
{
    SkiaPath::SkiaPath() = default;
    SkiaPath::~SkiaPath() = default;

    const SkPath& SkiaPath::GetSkPath() const
    {
        return mPath;
    }

    void SkiaPath::Construct ( const std::vector<DrawType>& aCommands )
    {
        Construct ( aCommands.data(), aCommands.size() );
    }

    void SkiaPath::Construct ( const DrawType* aCommands, size_t aCommandCount )
    {
        mPath.reset();
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
                last_move = last_point = ( ( cmd == 'm' ) ? last_point : Vector2{0, 0} ) + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                mPath.moveTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                i += 2;
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point += {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 2;
                }
            }
            break;
            case 'Z':
            case 'z':
                mPath.close();
                last_point = last_move;
                break;
            case 'L':
            case 'l':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point = ( ( cmd == 'l' ) ? last_point : Vector2{0, 0} ) + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 2;
                }
                break;
            case 'H':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point[0] = std::get<double> ( *i );
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    ++i;
                }
                break;
            case 'h':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point[0] += std::get<double> ( *i );
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    ++i;
                }
                break;
            case 'V':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point[1] = std::get<double> ( *i );
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    ++i;
                }
                break;
            case 'v':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_point[1] += std::get<double> ( *i );
                    mPath.lineTo ( static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    ++i;
                }
                break;
            case 'C':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    double x1 = std::get<double> ( * ( i ) );
                    double y1 = std::get<double> ( * ( i + 1 ) );
                    last_c_ctrl = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    last_point  = {std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) };
                    mPath.cubicTo ( static_cast<SkScalar> ( x1 ), static_cast<SkScalar> ( y1 ),
                                    static_cast<SkScalar> ( last_c_ctrl[0] ), static_cast<SkScalar> ( last_c_ctrl[1] ),
                                    static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 6;
                }
                break;
            case 'c':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    double x1 = last_point[0] + std::get<double> ( *i );
                    double y1 = last_point[1] + std::get<double> ( * ( i + 1 ) );
                    last_c_ctrl = last_point + Vector2{std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    last_point += {std::get<double> ( * ( i + 4 ) ), std::get<double> ( * ( i + 5 ) ) };
                    mPath.cubicTo ( static_cast<SkScalar> ( x1 ), static_cast<SkScalar> ( y1 ),
                                    static_cast<SkScalar> ( last_c_ctrl[0] ), static_cast<SkScalar> ( last_c_ctrl[1] ),
                                    static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 6;
                }
                break;
            case 'S':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    Vector2 cp1 = ( last_cmd == 'C' || last_cmd == 'S' ) ? Vector2{ ( 2 * last_point[0] ) - last_c_ctrl[0], ( 2 * last_point[1] ) - last_c_ctrl[1]} :
                                  last_point;
                    last_c_ctrl = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    last_point = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPath.cubicTo ( static_cast<SkScalar> ( cp1[0] ), static_cast<SkScalar> ( cp1[1] ),
                                    static_cast<SkScalar> ( last_c_ctrl[0] ), static_cast<SkScalar> ( last_c_ctrl[1] ),
                                    static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 4;
                }
                break;
            case 's':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    Vector2 cp1 = ( last_cmd == 'c' || last_cmd == 's' ) ? Vector2{ ( 2 * last_point[0] ) - last_c_ctrl[0], ( 2 * last_point[1] ) - last_c_ctrl[1]} :
                                  last_point;
                    last_c_ctrl = last_point + Vector2{std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    last_point += {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPath.cubicTo ( static_cast<SkScalar> ( cp1[0] ), static_cast<SkScalar> ( cp1[1] ),
                                    static_cast<SkScalar> ( last_c_ctrl[0] ), static_cast<SkScalar> ( last_c_ctrl[1] ),
                                    static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 4;
                }
                break;
            case 'Q':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = {std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    last_point = {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPath.quadTo ( static_cast<SkScalar> ( last_q_ctrl[0] ), static_cast<SkScalar> ( last_q_ctrl[1] ),
                                   static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 4;
                }
                break;
            case 'q':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = last_point + Vector2{std::get<double> ( *i ), std::get<double> ( * ( i + 1 ) ) };
                    last_point += {std::get<double> ( * ( i + 2 ) ), std::get<double> ( * ( i + 3 ) ) };
                    mPath.quadTo ( static_cast<SkScalar> ( last_q_ctrl[0] ), static_cast<SkScalar> ( last_q_ctrl[1] ),
                                   static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 4;
                }
                break;
            case 'T':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = ( last_cmd == 'Q' || last_cmd == 'T' ) ? Vector2{ ( 2 * last_point[0] ) - last_q_ctrl[0], ( 2 * last_point[1] ) - last_q_ctrl[1]} :
                                  last_point;
                    last_point = {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    mPath.quadTo ( static_cast<SkScalar> ( last_q_ctrl[0] ), static_cast<SkScalar> ( last_q_ctrl[1] ),
                                   static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 2;
                }
                break;
            case 't':
                while ( i != end && std::holds_alternative<double> ( *i ) )
                {
                    last_q_ctrl = ( last_cmd == 'q' || last_cmd == 't' ) ? Vector2{ ( 2 * last_point[0] ) - last_q_ctrl[0], ( 2 * last_point[1] ) - last_q_ctrl[1]} :
                                  last_point;
                    last_point += {std::get<double> ( * ( i ) ), std::get<double> ( * ( i + 1 ) ) };
                    mPath.quadTo ( static_cast<SkScalar> ( last_q_ctrl[0] ), static_cast<SkScalar> ( last_q_ctrl[1] ),
                                   static_cast<SkScalar> ( last_point[0] ), static_cast<SkScalar> ( last_point[1] ) );
                    i += 2;
                }
                break;
            case 'A':
                while ( i != end && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    double rx = std::get<double> ( * ( i ) );
                    double ry = std::get<double> ( * ( i + 1 ) );
                    double rotation = std::get<double> ( * ( i + 2 ) );
                    bool largeArc = std::get<bool> ( * ( i + 3 ) );
                    bool sweep = std::get<bool> ( * ( i + 4 ) );
                    double x2 = std::get<double> ( * ( i + 5 ) );
                    double y2 = std::get<double> ( * ( i + 6 ) );
                    mPath.arcTo ( static_cast<SkScalar> ( rx ), static_cast<SkScalar> ( ry ),
                                  static_cast<SkScalar> ( rotation ),
                                  largeArc ? SkPath::kLarge_ArcSize : SkPath::kSmall_ArcSize,
                                  sweep ? SkPathDirection::kCW : SkPathDirection::kCCW,
                                  static_cast<SkScalar> ( x2 ), static_cast<SkScalar> ( y2 ) );
                    last_point = {x2, y2};
                    i += 7;
                }
                break;
            case 'a':
                while ( i != end && !std::holds_alternative<uint64_t> ( *i ) )
                {
                    double rx = std::get<double> ( * ( i ) );
                    double ry = std::get<double> ( * ( i + 1 ) );
                    double rotation = std::get<double> ( * ( i + 2 ) );
                    bool largeArc = std::get<bool> ( * ( i + 3 ) );
                    bool sweep = std::get<bool> ( * ( i + 4 ) );
                    double x2 = last_point[0] + std::get<double> ( * ( i + 5 ) );
                    double y2 = last_point[1] + std::get<double> ( * ( i + 6 ) );
                    mPath.arcTo ( static_cast<SkScalar> ( rx ), static_cast<SkScalar> ( ry ),
                                  static_cast<SkScalar> ( rotation ),
                                  largeArc ? SkPath::kLarge_ArcSize : SkPath::kSmall_ArcSize,
                                  sweep ? SkPathDirection::kCW : SkPathDirection::kCCW,
                                  static_cast<SkScalar> ( x2 ), static_cast<SkScalar> ( y2 ) );
                    last_point = {x2, y2};
                    i += 7;
                }
                break;
            }
            last_cmd = cmd;
        }
    }

    double SkiaPath::GetTotalLength() const
    {
        SkPathMeasure measure ( mPath, false );
        double total = 0.0;
        do
        {
            total += static_cast<double> ( measure.getLength() );
        }
        while ( measure.nextContour() );
        return total;
    }

    PathPoint SkiaPath::GetPointAtLength ( double aDistance ) const
    {
        SkPathMeasure measure ( mPath, false );
        SkScalar remaining = static_cast<SkScalar> ( aDistance );
        do
        {
            SkScalar contourLen = measure.getLength();
            if ( remaining <= contourLen )
            {
                SkPoint pos;
                SkVector tan;
                if ( measure.getPosTan ( remaining, &pos, &tan ) )
                {
                    return PathPoint{static_cast<double> ( pos.fX ),
                                     static_cast<double> ( pos.fY ),
                                     std::atan2 ( static_cast<double> ( tan.fY ), static_cast<double> ( tan.fX ) ) };
                }
            }
            remaining -= contourLen;
        }
        while ( measure.nextContour() );
        // Past the end — return the last point.
        SkPoint lastPt;
        if ( mPath.getLastPt ( &lastPt ) )
        {
            return PathPoint{static_cast<double> ( lastPt.fX ), static_cast<double> ( lastPt.fY ), 0.0};
        }
        return PathPoint{0.0, 0.0, 0.0};
    }

    bool SkiaPath::IsClosed() const
    {
        SkPath::Iter iter ( mPath, false );
        SkPoint pts[4];
        SkPath::Verb verb;
        while ( ( verb = iter.next ( pts ) ) != SkPath::kDone_Verb )
        {
            if ( verb == SkPath::kClose_Verb )
            {
                return true;
            }
        }
        return false;
    }
}
