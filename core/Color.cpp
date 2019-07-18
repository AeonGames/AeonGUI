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
#include <iostream>
#include <unordered_map>
namespace AeonGUI
{
    const std::regex Color::ColorRegex{"#[0-9A-Fa-f]{8}|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|aliceblue|antiquewhite|aqua|aquamarine|azure|beige|bisque|black|blanchedalmond|blue|blueviolet|brown|burlywood|cadetblue|chartreuse|chocolate|coral|cornflowerblue|cornsilk|crimson|cyan|darkblue|darkcyan|darkgoldenrod|darkgray|darkgreen|darkgrey|darkkhaki|darkmagenta|darkolivegreen|darkorange|darkorchid|darkred|darksalmon|darkseagreen|darkslateblue|darkslategray|darkslategrey|darkturquoise|darkviolet|deeppink|deepskyblue|dimgray|dimgrey|dodgerblue|firebrick|floralwhite|forestgreen|fuchsia|gainsboro|ghostwhite|gold|goldenrod|gray|grey|green|greenyellow|honeydew|hotpink|indianred|indigo|ivory|khaki|lavender|lavenderblush|lawngreen|lemonchiffon|lightblue|lightcoral|lightcyan|lightgoldenrodyellow|lightgray|lightgreen|lightgrey|lightpink|lightsalmon|lightseagreen|lightskyblue|lightslategray|lightslategrey|lightsteelblue|lightyellow|lime|limegreen|linen|magenta|maroon|mediumaquamarine|mediumblue|mediumorchid|mediumpurple|mediumseagreen|mediumslateblue|mediumspringgreen|mediumturquoise|mediumvioletred|midnightblue|mintcream|mistyrose|moccasin|navajowhite|navy|none|oldlace|olive|olivedrab|orange|orangered|orchid|palegoldenrod|palegreen|paleturquoise|palevioletred|papayawhip|peachpuff|peru|pink|plum|powderblue|purple|red|rosybrown|royalblue|saddlebrown|salmon|sandybrown|seagreen|seashell|sienna|silver|skyblue|slateblue|slategray|slategrey|snow|springgreen|steelblue|tan|teal|thistle|tomato|transparent|turquoise|violet|wheat|white|whitesmoke|yellow|yellowgreen"};
    static const std::unordered_map<std::string, uint32_t> color_map
    {
        {"aliceblue", 0xfff0f8ff},
        {"antiquewhite", 0xfffaebd7},
        {"aqua", 0xff00ffff},
        {"aquamarine", 0xff7fffd4},
        {"azure", 0xfff0ffff},
        {"beige", 0xfff5f5dc},
        {"bisque", 0xffffe4c4},
        {"black", 0xff000000},
        {"blanchedalmond", 0xffffebcd},
        {"blue", 0xff0000ff},
        {"blueviolet", 0xff8a2be2},
        {"brown", 0xffa52a2a},
        {"burlywood", 0xffdeb887},
        {"cadetblue", 0xff5f9ea0},
        {"chartreuse", 0xff7fff00},
        {"chocolate", 0xffd2691e},
        {"coral", 0xffff7f50},
        {"cornflowerblue", 0xff6495ed},
        {"cornsilk", 0xfffff8dc},
        {"crimson", 0xffdc143c},
        {"cyan", 0xff00ffff},
        {"darkblue", 0xff00008b},
        {"darkcyan", 0xff008b8b},
        {"darkgoldenrod", 0xffb8860b},
        {"darkgray", 0xffa9a9a9},
        {"darkgreen", 0xff006400},
        {"darkgrey", 0xffa9a9a9},
        {"darkkhaki", 0xffbdb76b},
        {"darkmagenta", 0xff8b008b},
        {"darkolivegreen", 0xff556b2f},
        {"darkorange", 0xffff8c00},
        {"darkorchid", 0xff9932cc},
        {"darkred", 0xff8b0000},
        {"darksalmon", 0xffe9967a},
        {"darkseagreen", 0xff8fbc8f},
        {"darkslateblue", 0xff483d8b},
        {"darkslategray", 0xff2f4f4f},
        {"darkslategrey", 0xff2f4f4f},
        {"darkturquoise", 0xff00ced1},
        {"darkviolet", 0xff9400d3},
        {"deeppink", 0xffff1493},
        {"deepskyblue", 0xff00bfff},
        {"dimgray", 0xff696969},
        {"dimgrey", 0xff696969},
        {"dodgerblue", 0xff1e90ff},
        {"firebrick", 0xffb22222},
        {"floralwhite", 0xfffffaf0},
        {"forestgreen", 0xff228b22},
        {"fuchsia", 0xffff00ff},
        {"gainsboro", 0xffdcdcdc},
        {"ghostwhite", 0xfff8f8ff},
        {"gold", 0xffffd700},
        {"goldenrod", 0xffdaa520},
        {"gray", 0xff808080},
        {"grey", 0xff808080},
        {"green", 0xff008000},
        {"greenyellow", 0xffadff2f},
        {"honeydew", 0xfff0fff0},
        {"hotpink", 0xffff69b4},
        {"indianred", 0xffcd5c5c},
        {"indigo", 0xff4b0082},
        {"ivory", 0xfffffff0},
        {"khaki", 0xfff0e68c},
        {"lavender", 0xffe6e6fa},
        {"lavenderblush", 0xfffff0f5},
        {"lawngreen", 0xff7cfc00},
        {"lemonchiffon", 0xfffffacd},
        {"lightblue", 0xffadd8e6},
        {"lightcoral", 0xfff08080},
        {"lightcyan", 0xffe0ffff},
        {"lightgoldenrodyellow", 0xfffafad2},
        {"lightgray", 0xffd3d3d3},
        {"lightgreen", 0xff90ee90},
        {"lightgrey", 0xffd3d3d3},
        {"lightpink", 0xffffb6c1},
        {"lightsalmon", 0xffffa07a},
        {"lightseagreen", 0xff20b2aa},
        {"lightskyblue", 0xff87cefa},
        {"lightslategray", 0xff778899},
        {"lightslategrey", 0xff778899},
        {"lightsteelblue", 0xffb0c4de},
        {"lightyellow", 0xffffffe0},
        {"lime", 0xff00ff00},
        {"limegreen", 0xff32cd32},
        {"linen", 0xfffaf0e6},
        {"magenta", 0xffff00ff},
        {"maroon", 0xff800000},
        {"mediumaquamarine", 0xff66cdaa},
        {"mediumblue", 0xff0000cd},
        {"mediumorchid", 0xffba55d3},
        {"mediumpurple", 0xff9370db},
        {"mediumseagreen", 0xff3cb371},
        {"mediumslateblue", 0xff7b68ee},
        {"mediumspringgreen", 0xff00fa9a},
        {"mediumturquoise", 0xff48d1cc},
        {"mediumvioletred", 0xffc71585},
        {"midnightblue", 0xff191970},
        {"mintcream", 0xfff5fffa},
        {"mistyrose", 0xffffe4e1},
        {"moccasin", 0xffffe4b5},
        {"navajowhite", 0xffffdead},
        {"navy", 0xff000080},
        {"none", 0x00000000}, //<-- Look at me I'm Special.
        {"oldlace", 0xfffdf5e6},
        {"olive", 0xff808000},
        {"olivedrab", 0xff6b8e23},
        {"orange", 0xffffa500},
        {"orangered", 0xffff4500},
        {"orchid", 0xffda70d6},
        {"palegoldenrod", 0xffeee8aa},
        {"palegreen", 0xff98fb98},
        {"paleturquoise", 0xffafeeee},
        {"palevioletred", 0xffdb7093},
        {"papayawhip", 0xffffefd5},
        {"peachpuff", 0xffffdab9},
        {"peru", 0xffcd853f},
        {"pink", 0xffffc0cb},
        {"plum", 0xffdda0dd},
        {"powderblue", 0xffb0e0e6},
        {"purple", 0xff800080},
        {"red", 0xffff0000},
        {"rosybrown", 0xffbc8f8f},
        {"royalblue", 0xff4169e1},
        {"saddlebrown", 0xff8b4513},
        {"salmon", 0xfffa8072},
        {"sandybrown", 0xfff4a460},
        {"seagreen", 0xff2e8b57},
        {"seashell", 0xfffff5ee},
        {"sienna", 0xffa0522d},
        {"silver", 0xffc0c0c0},
        {"skyblue", 0xff87ceeb},
        {"slateblue", 0xff6a5acd},
        {"slategray", 0xff708090},
        {"slategrey", 0xff708090},
        {"snow", 0xfffffafa},
        {"springgreen", 0xff00ff7f},
        {"steelblue", 0xff4682b4},
        {"tan", 0xffd2b48c},
        {"teal", 0xff008080},
        {"thistle", 0xffd8bfd8},
        {"tomato", 0xffff6347},
        {"transparent", 0x00000000}, //<-- Look at me I'm Special.
        {"turquoise", 0xff40e0d0},
        {"violet", 0xffee82ee},
        {"wheat", 0xfff5deb3},
        {"white", 0xffffffff},
        {"whitesmoke", 0xfff5f5f5},
        {"yellow", 0xffffff00},
        {"yellowgreen", 0xff9acd32},
    };

    static const std::regex color_regex3{"#([0-9A-Fa-f])([0-9A-Fa-f])([0-9A-Fa-f])"};
    static const std::regex color_regex6{"#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"};
    static const std::regex color_regex8{"#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"};

    Color::Color() : bgra ( 0 ) {}
    Color::Color ( uint32_t value ) : bgra ( value ) {}
    Color::Color ( uint8_t A, uint8_t R, uint8_t G, uint8_t B )
        : b ( B ), g ( G ), r ( R ), a ( A ) {}

    Color::Color ( const std::string& value )
    {
        std::smatch color_match;
        std::unordered_map<std::string, uint32_t>::const_iterator color_iterator;
        if ( std::regex_search ( value, color_match, color_regex8 ) )
        {
            a = static_cast<uint8_t> ( std::stoul ( color_match[1].str(), nullptr, 16 ) );
            r = static_cast<uint8_t> ( std::stoul ( color_match[2].str(), nullptr, 16 ) );
            g = static_cast<uint8_t> ( std::stoul ( color_match[3].str(), nullptr, 16 ) );
            b = static_cast<uint8_t> ( std::stoul ( color_match[4].str(), nullptr, 16 ) );
        }
        else if ( std::regex_search ( value, color_match, color_regex6 ) )
        {
            a = 255;
            r = static_cast<uint8_t> ( std::stoul ( color_match[1].str(), nullptr, 16 ) );
            g = static_cast<uint8_t> ( std::stoul ( color_match[2].str(), nullptr, 16 ) );
            b = static_cast<uint8_t> ( std::stoul ( color_match[3].str(), nullptr, 16 ) );
        }
        else if ( std::regex_search ( value, color_match, color_regex3 ) )
        {
            a = 255;
            r = static_cast<uint8_t> ( std::stoul ( color_match[1].str() + color_match[1].str(), nullptr, 16 ) );
            g = static_cast<uint8_t> ( std::stoul ( color_match[2].str() + color_match[2].str(), nullptr, 16 ) );
            b = static_cast<uint8_t> ( std::stoul ( color_match[3].str() + color_match[3].str(), nullptr, 16 ) );
        }
        else if ( ( color_iterator = color_map.find ( value ) ) != color_map.end() )
        {
            bgra = color_iterator->second;
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
