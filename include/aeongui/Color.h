/*
Copyright (C) 2010-2012,2019,2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_COLOR_H
#define AEONGUI_COLOR_H
#include <string>
#include <cstdint>
#include <regex>
#include <variant>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    /// W3C CSS3 color keyword names (see https://en.wikipedia.org/wiki/Web_colors)
    enum CSS3Color : uint32_t
    {
        aliceblue = 0xfff0f8ff,
        antiquewhite = 0xfffaebd7,
        aqua = 0xff00ffff,
        aquamarine = 0xff7fffd4,
        azure = 0xfff0ffff,
        beige = 0xfff5f5dc,
        bisque = 0xffffe4c4,
        black = 0xff000000,
        blanchedalmond = 0xffffebcd,
        blue = 0xff0000ff,
        blueviolet = 0xff8a2be2,
        brown = 0xffa52a2a,
        burlywood = 0xffdeb887,
        cadetblue = 0xff5f9ea0,
        chartreuse = 0xff7fff00,
        chocolate = 0xffd2691e,
        coral = 0xffff7f50,
        cornflowerblue = 0xff6495ed,
        cornsilk = 0xfffff8dc,
        crimson = 0xffdc143c,
        cyan = 0xff00ffff,
        darkblue = 0xff00008b,
        darkcyan = 0xff008b8b,
        darkgoldenrod = 0xffb8860b,
        darkgray = 0xffa9a9a9,
        darkgreen = 0xff006400,
        darkgrey = 0xffa9a9a9,
        darkkhaki = 0xffbdb76b,
        darkmagenta = 0xff8b008b,
        darkolivegreen = 0xff556b2f,
        darkorange = 0xffff8c00,
        darkorchid = 0xff9932cc,
        darkred = 0xff8b0000,
        darksalmon = 0xffe9967a,
        darkseagreen = 0xff8fbc8f,
        darkslateblue = 0xff483d8b,
        darkslategray = 0xff2f4f4f,
        darkslategrey = 0xff2f4f4f,
        darkturquoise = 0xff00ced1,
        darkviolet = 0xff9400d3,
        deeppink = 0xffff1493,
        deepskyblue = 0xff00bfff,
        dimgray = 0xff696969,
        dimgrey = 0xff696969,
        dodgerblue = 0xff1e90ff,
        firebrick = 0xffb22222,
        floralwhite = 0xfffffaf0,
        forestgreen = 0xff228b22,
        fuchsia = 0xffff00ff,
        gainsboro = 0xffdcdcdc,
        ghostwhite = 0xfff8f8ff,
        gold = 0xffffd700,
        goldenrod = 0xffdaa520,
        gray = 0xff808080,
        grey = 0xff808080,
        green = 0xff008000,
        greenyellow = 0xffadff2f,
        honeydew = 0xfff0fff0,
        hotpink = 0xffff69b4,
        indianred = 0xffcd5c5c,
        indigo = 0xff4b0082,
        ivory = 0xfffffff0,
        khaki = 0xfff0e68c,
        lavender = 0xffe6e6fa,
        lavenderblush = 0xfffff0f5,
        lawngreen = 0xff7cfc00,
        lemonchiffon = 0xfffffacd,
        lightblue = 0xffadd8e6,
        lightcoral = 0xfff08080,
        lightcyan = 0xffe0ffff,
        lightgoldenrodyellow = 0xfffafad2,
        lightgray = 0xffd3d3d3,
        lightgreen = 0xff90ee90,
        lightgrey = 0xffd3d3d3,
        lightpink = 0xffffb6c1,
        lightsalmon = 0xffffa07a,
        lightseagreen = 0xff20b2aa,
        lightskyblue = 0xff87cefa,
        lightslategray = 0xff778899,
        lightslategrey = 0xff778899,
        lightsteelblue = 0xffb0c4de,
        lightyellow = 0xffffffe0,
        lime = 0xff00ff00,
        limegreen = 0xff32cd32,
        linen = 0xfffaf0e6,
        magenta = 0xffff00ff,
        maroon = 0xff800000,
        mediumaquamarine = 0xff66cdaa,
        mediumblue = 0xff0000cd,
        mediumorchid = 0xffba55d3,
        mediumpurple = 0xff9370db,
        mediumseagreen = 0xff3cb371,
        mediumslateblue = 0xff7b68ee,
        mediumspringgreen = 0xff00fa9a,
        mediumturquoise = 0xff48d1cc,
        mediumvioletred = 0xffc71585,
        midnightblue = 0xff191970,
        mintcream = 0xfff5fffa,
        mistyrose = 0xffffe4e1,
        moccasin = 0xffffe4b5,
        navajowhite = 0xffffdead,
        navy = 0xff000080,
        oldlace = 0xfffdf5e6,
        olive = 0xff808000,
        olivedrab = 0xff6b8e23,
        orange = 0xffffa500,
        orangered = 0xffff4500,
        orchid = 0xffda70d6,
        palegoldenrod = 0xffeee8aa,
        palegreen = 0xff98fb98,
        paleturquoise = 0xffafeeee,
        palevioletred = 0xffdb7093,
        papayawhip = 0xffffefd5,
        peachpuff = 0xffffdab9,
        peru = 0xffcd853f,
        pink = 0xffffc0cb,
        plum = 0xffdda0dd,
        powderblue = 0xffb0e0e6,
        purple = 0xff800080,
        red = 0xffff0000,
        rosybrown = 0xffbc8f8f,
        royalblue = 0xff4169e1,
        saddlebrown = 0xff8b4513,
        salmon = 0xfffa8072,
        sandybrown = 0xfff4a460,
        seagreen = 0xff2e8b57,
        seashell = 0xfffff5ee,
        sienna = 0xffa0522d,
        silver = 0xffc0c0c0,
        skyblue = 0xff87ceeb,
        slateblue = 0xff6a5acd,
        slategray = 0xff708090,
        slategrey = 0xff708090,
        snow = 0xfffffafa,
        springgreen = 0xff00ff7f,
        steelblue = 0xff4682b4,
        tan = 0xffd2b48c,
        teal = 0xff008080,
        thistle = 0xffd8bfd8,
        tomato = 0xffff6347,
        transparent = 0x00000000, ///< Special color case
        turquoise = 0xff40e0d0,
        violet = 0xffee82ee,
        wheat = 0xfff5deb3,
        white = 0xffffffff,
        whitesmoke = 0xfff5f5f5,
        yellow = 0xffffff00,
        yellowgreen = 0xff9acd32
    };
    /** \brief Pixel Color Union.
        The Color union allows access to each unsigned 8 bit RGBA color component individually or as a single 32 bit unsigned integer,
        making possible color asignments similar to those used on the web (IE: color = 0xFF000000 with the format 0xAARRGGBB).
        The union type is also 32bits wide which makes it suitable to be passed as value or as reference.
        Wherever a Color variable can be passed it \a should be possible to pass a 32 bit unsigned integer value, but some compilers may complain depending on the set warning level.
    */
    union Color
    {
        DLL static bool IsColor ( const std::string& value, uint32_t* color_value = nullptr );
        DLL static const std::regex ColorRegex;
        DLL Color();
        /*! \brief 32 bit Unsigned integer constructor.
            \param value 32 bit color value.
        */
        DLL explicit Color ( uint32_t value );

        DLL Color ( const std::string& value );
        /*! \brief 4 8 bit Unsigned integer component constructor.
            \param A Alpha color value.
            \param R Red color value.
            \param G Green color value.
            \param B Blue color value.
        */
        DLL Color ( uint8_t A, uint8_t R, uint8_t G, uint8_t B );

        /*! \brief Blends the color with the incoming source color.
            To better handle transparency the blend function uses the source alpha as source factor
            and one minus source alpha as destination factor.
            \param src Incomming source color.*/
        DLL void Blend ( Color src );

        DLL double R() const;
        DLL double G() const;
        DLL double B() const;
        DLL double A() const;
        DLL std::string ToString() const;
        uint32_t bgra; ///< 32 bit Unsigned integer color value.
        struct
        {
#if 0
            uint8_t r;
            uint8_t g;
            uint8_t b;
            uint8_t a;
#else
            uint8_t b; ///< Blue color element.
            uint8_t g; ///< Green color element.
            uint8_t r; ///< Red color element.
            uint8_t a; ///< Alpha color element.
#endif
        };
    };
    /// Alias monostate to none.
    using none = std::monostate;
    /// A special color type that distinguishes when no color is set.
    using ColorAttr = std::variant<none, Color>;
}
#endif