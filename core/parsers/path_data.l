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
#include "aeongui/DrawType.h"
#include "../core/parsers/dstype.h"
#include "path_data_parser.hpp"
#define YY_NO_UNISTD_H 1
%}

%option nounput
%option nounistd
%option never-interactive
%option noreject
%option nostdinit
%option noyywrap
%option noyylineno
%option nodebug
%option prefix="d"

number               -?[0-9]*\.?[0-9]+([eE][-+][0-9]+)?

%%
{number}                {
							dlval = std::strtod(yytext,nullptr);
							return NUMBER;
						}
"M"                     { dlval = static_cast<uint64_t>('M'); return 'M';}
"m"                     { dlval = static_cast<uint64_t>('m'); return 'm';}
"Z"                     { dlval = static_cast<uint64_t>('Z'); return 'Z';}
"z"                     { dlval = static_cast<uint64_t>('z'); return 'z';}
"L"                     { dlval = static_cast<uint64_t>('L'); return 'L';}
"l"                     { dlval = static_cast<uint64_t>('l'); return 'l';}
"H"                     { dlval = static_cast<uint64_t>('H'); return 'H';}
"h"                     { dlval = static_cast<uint64_t>('h'); return 'h';}
"V"                     { dlval = static_cast<uint64_t>('V'); return 'V';}
"v"                     { dlval = static_cast<uint64_t>('v'); return 'v';}
"C"                     { dlval = static_cast<uint64_t>('C'); return 'C';}
"c"                     { dlval = static_cast<uint64_t>('c'); return 'c';}
"S"                     { dlval = static_cast<uint64_t>('S'); return 'S';}
"s"                     { dlval = static_cast<uint64_t>('s'); return 's';}
"Q"                     { dlval = static_cast<uint64_t>('Q'); return 'Q';}
"q"                     { dlval = static_cast<uint64_t>('q'); return 'q';}
"T"                     { dlval = static_cast<uint64_t>('T'); return 'T';}
"t"                     { dlval = static_cast<uint64_t>('t'); return 't';}
"A"                     { dlval = static_cast<uint64_t>('A'); return 'A';}
"a"                     { dlval = static_cast<uint64_t>('a'); return 'a';}
.
%%

namespace AeonGUI
{
	namespace DOM
	{
		int ParsePathData(std::vector<DrawType>& aPath,const char* string)
		{
			YY_BUFFER_STATE state=d_scan_string(string);
			d_switch_to_buffer(state);
			BEGIN(INITIAL);
			return dparse(aPath);
		}
	}
}
