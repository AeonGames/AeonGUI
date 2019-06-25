%{
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
#include <cstddef>
#include <cstdint>
#include "path_data_parser.hpp"
%}

%option nounput
%option noreject
%option nostdinit
%option noyywrap
%option noyylineno
%option nodebug
%option prefix="d"

number               -?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?)

%%
{number}                return NUMBER;
","                     return ',';
"M"                     return 'M';
"m"                     return 'm';
"Z"                     return 'Z';
"z"                     return 'z';
"L"                     return 'L';
"l"                     return 'l';
"H"                     return 'H';
"h"                     return 'h';
"V"                     return 'V';
"v"                     return 'v';
"C"                     return 'C';
"c"                     return 'c';
"S"                     return 'S';
"s"                     return 's';
"Q"                     return 'Q';
"q"                     return 'q';
"T"                     return 'T';
"t"                     return 't';
"A"                     return 'A';
"a"                     return 'a';
.
%%

namespace AeonGUI
{
	int ParsePathData(const char* string)
	{
		YY_BUFFER_STATE state=d_scan_string(string);
		d_switch_to_buffer(state);
		BEGIN(INITIAL);
		return dparse();
	}
}