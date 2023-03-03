%{
/*
Copyright (C) 2019-2021,2023 Rodrigo Jose Hernandez Cordoba

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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "aeongui/AttributeMap.h"
#include "../core/parsers/stylestype.h"
#include "style_parser.hpp"
#define YY_NO_UNISTD_H 1
%}

%option nounput
%option nounistd
%option never-interactive
%option noreject
%option nostdinit
%option nodebug
%option noyywrap
%option noyylineno
%option prefix="style"

h               [0-9A-Fa-f]
nmstart		    [_a-z]
nmchar		    [_a-z0-9-]
ident		    -?[_a-z][_a-z0-9-]*
number          -?[0-9]*\.?[0-9]+([eE][-+][0-9]+)?
delim           ;|:
				/* Keep color list in strict alphanumerical order */
color           #[0-9A-Fa-f]{8}|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|aliceblue|antiquewhite|aqua|aquamarine|azure|beige|bisque|black|blanchedalmond|blue|blueviolet|brown|burlywood|cadetblue|chartreuse|chocolate|coral|cornflowerblue|cornsilk|crimson|cyan|darkblue|darkcyan|darkgoldenrod|darkgray|darkgreen|darkgrey|darkkhaki|darkmagenta|darkolivegreen|darkorange|darkorchid|darkred|darksalmon|darkseagreen|darkslateblue|darkslategray|darkslategrey|darkturquoise|darkviolet|deeppink|deepskyblue|dimgray|dimgrey|dodgerblue|firebrick|floralwhite|forestgreen|fuchsia|gainsboro|ghostwhite|gold|goldenrod|gray|grey|green|greenyellow|honeydew|hotpink|indianred|indigo|ivory|khaki|lavender|lavenderblush|lawngreen|lemonchiffon|lightblue|lightcoral|lightcyan|lightgoldenrodyellow|lightgray|lightgreen|lightgrey|lightpink|lightsalmon|lightseagreen|lightskyblue|lightslategray|lightslategrey|lightsteelblue|lightyellow|lime|limegreen|linen|magenta|maroon|mediumaquamarine|mediumblue|mediumorchid|mediumpurple|mediumseagreen|mediumslateblue|mediumspringgreen|mediumturquoise|mediumvioletred|midnightblue|mintcream|mistyrose|moccasin|navajowhite|navy|oldlace|olive|olivedrab|orange|orangered|orchid|palegoldenrod|palegreen|paleturquoise|palevioletred|papayawhip|peachpuff|peru|pink|plum|powderblue|purple|red|rosybrown|royalblue|saddlebrown|salmon|sandybrown|seagreen|seashell|sienna|silver|skyblue|slateblue|slategray|slategrey|snow|springgreen|steelblue|tan|teal|thistle|tomato|transparent|turquoise|violet|wheat|white|whitesmoke|yellow|yellowgreen

%%

{color}        		  	{
							stylelval = AeonGUI::ColorAttr{AeonGUI::Color{yytext}};
							return COLOR;
						}
none        		  	{
							stylelval = AeonGUI::ColorAttr{};
							return COLOR;
						}
{ident}			    	{
							stylelval = yytext;
							return IDENT;
						}
{number}				{
							stylelval = std::strtod(yytext,nullptr);
							return NUMBER;
						}
{delim}					{
							return yytext[0];
						}
.

%%

namespace AeonGUI
{
	int ParseStyle(AttributeMap& aStyle,const char* string)
	{
		YY_BUFFER_STATE state=style_scan_string(string);
		style_switch_to_buffer(state);
		BEGIN(INITIAL);
		return styleparse(aStyle);
	}
}
