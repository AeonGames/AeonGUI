/******************************************************************************
Copyright (C) 2010-2012,2019 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include "aeongui/Color.h"
#include <algorithm>
#include <regex>
#include <iostream>
namespace AeonGUI
{
    static const std::regex color_regex{"#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"};

    Color::Color() : bgra ( 0 ) {}
    Color::Color ( uint32_t value ) : bgra ( value ) {}
    Color::Color ( uint8_t A, uint8_t R, uint8_t G, uint8_t B )
        : b ( B ), g ( G ), r ( R ), a ( A ) {}

    Color::Color ( const std::string& value )
    {
        std::smatch color_match;
        if ( std::regex_search ( value, color_match, color_regex ) )
        {
            a = 255;
            r = std::stoul ( color_match[1].str(), nullptr, 16 );
            g = std::stoul ( color_match[2].str(), nullptr, 16 );
            b = std::stoul ( color_match[3].str(), nullptr, 16 );
        }
    }

    double Color::R() const
    {
        return static_cast<double> ( r ) / 255.0;
    }
    double Color::G() const
    {
        return static_cast<double> ( g ) / 255.0;
    }
    double Color::B() const
    {
        return static_cast<double> ( b ) / 255.0;
    }
    double Color::A() const
    {
        return static_cast<double> ( a ) / 255.0;
    }

    void Color::Blend ( Color src )
    {
        if ( ( src.a == 0 ) )
        {
            /*  If the source alpha is 0
                the destination color is unchanged */
            return;
        }
        else if ( ( src.a == 255 ) )
        {
            /*  Full source opacity
                do a simple replacement*/
            bgra = src.bgra;
        }
        else
        {
            float sfactor = ( static_cast<float> ( src.a ) / 255.0f );
            float dfactor = 1.0f - sfactor;
            r = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.r ) * sfactor + static_cast<float> ( r ) * dfactor ) ) );
            g = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.g ) * sfactor + static_cast<float> ( g ) * dfactor ) ) );
            b = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.b ) * sfactor + static_cast<float> ( b ) * dfactor ) ) );
            // Just acumulate alpha, if something looks odd try multipling the second addend by dfactor.
            a = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.a ) + static_cast<float> ( a ) ) ) );
        }
    }
}
