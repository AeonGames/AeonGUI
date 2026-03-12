/*
Copyright (C) 2019,2024-2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Matrix2x3.hpp"
#include "aeongui/Vector2.hpp"
#include <cmath>
#include <cstring>
#include <regex>
#include <vector>
#include <string>
namespace AeonGUI
{
    Matrix2x3::Matrix2x3()
    {
        mMatrix2x3[0] = mMatrix2x3[3] = 1.0f;
        mMatrix2x3[1] = mMatrix2x3[2] = mMatrix2x3[4] = mMatrix2x3[5] = 0.0f;
    }

    Matrix2x3::Matrix2x3 (
        double xx, double yx,
        double xy, double yy,
        double x0, double y0 )
    {
        mMatrix2x3[0] = xx;
        mMatrix2x3[1] = yx;
        mMatrix2x3[2] = xy;
        mMatrix2x3[3] = yy;
        mMatrix2x3[4] = x0;
        mMatrix2x3[5] = y0;
    }

    Matrix2x3::Matrix2x3 ( const std::array<const double, 6> aMatrixArray )
    {
        memcpy ( mMatrix2x3, aMatrixArray.data(), sizeof ( double ) * 6 );
    }

    Matrix2x3::Matrix2x3 ( double aRotation )
    {
        double radians = aRotation * ( M_PI / 180.0 ) ;
        mMatrix2x3[0] = mMatrix2x3[3] = std::cos ( radians );
        mMatrix2x3[1] = -std::sin ( radians );
        mMatrix2x3[2] = std::sin ( radians );
        mMatrix2x3[4] = mMatrix2x3[5] = 0.0;
    }
    Matrix2x3::Matrix2x3 ( const Vector2& aScale )
    {
        mMatrix2x3[0] = aScale[0];
        mMatrix2x3[3] = aScale[1];
        mMatrix2x3[1] = mMatrix2x3[2] = mMatrix2x3[4] = mMatrix2x3[5] = 0.0f;
    }
    Matrix2x3::Matrix2x3 ( const Vector2& aScale, double aRotation ) : Matrix2x3{aScale}
    {
        Matrix2x3 rotation ( aRotation );
        *this *= rotation;
    }
    Matrix2x3::Matrix2x3 ( const Vector2& aScale, double aRotation, const Vector2& aTranslation ) : Matrix2x3{aScale, aRotation}
    {
        mMatrix2x3[4] = aTranslation[0];
        mMatrix2x3[5] = aTranslation[1];
    }

    Matrix2x3& Matrix2x3::operator*= ( const Matrix2x3& aRight )
    {
        const Matrix2x3 local{*this};
        mMatrix2x3[0] = local[0] * aRight[0] + local[1] * aRight[2];
        mMatrix2x3[1] = local[0] * aRight[1] + local[1] * aRight[3];
        mMatrix2x3[2] = local[2] * aRight[0] + local[3] * aRight[2];
        mMatrix2x3[3] = local[2] * aRight[1] + local[3] * aRight[3];

        mMatrix2x3[4] = local[4] * aRight[0] + local[5] * aRight[2] + aRight[4];
        mMatrix2x3[5] = local[4] * aRight[1] + local[5] * aRight[3] + aRight[5];
        return *this;
    }

    const double& Matrix2x3::operator[] ( size_t aIndex ) const
    {
        return mMatrix2x3[aIndex];
    }

    const Matrix2x3 Abs ( const Matrix2x3& aMatrix2x3 )
    {
        return Matrix2x3
        {
            {
                std::abs ( aMatrix2x3[0] ),
                std::abs ( aMatrix2x3[1] ),
                std::abs ( aMatrix2x3[2] ),
                std::abs ( aMatrix2x3[3] ),
                std::abs ( aMatrix2x3[4] ),
                std::abs ( aMatrix2x3[5] ),
            }
        };
    }

    Matrix2x3 ParseSVGTransform ( const std::string& value )
    {
        Matrix2x3 result{};
        static const std::regex transform_regex ( R"((\w+)\s*\(([^)]+)\))" );
        static const std::regex num_regex ( R"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)" );
        auto begin = std::sregex_iterator ( value.begin(), value.end(), transform_regex );
        auto end = std::sregex_iterator();
        for ( auto it = begin; it != end; ++it )
        {
            std::string func = ( *it ) [1].str();
            std::string args_str = ( *it ) [2].str();
            std::vector<double> args;
            auto nbegin = std::sregex_iterator ( args_str.begin(), args_str.end(), num_regex );
            auto nend = std::sregex_iterator();
            for ( auto nit = nbegin; nit != nend; ++nit )
            {
                args.push_back ( std::stod ( nit->str() ) );
            }
            if ( func == "matrix" && args.size() == 6 )
            {
                result *= Matrix2x3 { args[0], args[1], args[2], args[3], args[4], args[5] };
            }
            else if ( func == "translate" && !args.empty() )
            {
                double tx = args[0];
                double ty = args.size() > 1 ? args[1] : 0.0;
                result *= Matrix2x3 { 1.0, 0.0, 0.0, 1.0, tx, ty };
            }
            else if ( func == "scale" && !args.empty() )
            {
                double sx = args[0];
                double sy = args.size() > 1 ? args[1] : sx;
                result *= Matrix2x3 { sx, 0.0, 0.0, sy, 0.0, 0.0 };
            }
            else if ( func == "rotate" && !args.empty() )
            {
                double angle = args[0] * M_PI / 180.0;
                double cx = args.size() > 1 ? args[1] : 0.0;
                double cy = args.size() > 2 ? args[2] : 0.0;
                double cos_a = std::cos ( angle );
                double sin_a = std::sin ( angle );
                result *= Matrix2x3 { cos_a, sin_a, -sin_a, cos_a,
                                      cx - cx * cos_a + cy * sin_a,
                                      cy - cx * sin_a - cy * cos_a };
            }
            else if ( func == "skewX" && !args.empty() )
            {
                result *= Matrix2x3 { 1.0, 0.0, std::tan ( args[0] * M_PI / 180.0 ), 1.0, 0.0, 0.0 };
            }
            else if ( func == "skewY" && !args.empty() )
            {
                result *= Matrix2x3 { 1.0, std::tan ( args[0] * M_PI / 180.0 ), 0.0, 1.0, 0.0, 0.0 };
            }
        }
        return result;
    }
}
