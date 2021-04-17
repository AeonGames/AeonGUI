%{
/*
Copyright (C) 2021 Rodrigo Jose Hernandez Cordoba

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

/* Interpreted as described in https://heycam.github.io/webidl/#idl-grammar */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "webidl_parser.hpp"
#define YY_NO_UNISTD_H 1
%}

%option nounput
%option nounistd
%option noreject
%option nostdinit
%option noyywrap
%option yylineno
%option never-interactive
%option prefix="webidl"

integer                 -?([1-9][0-9]*|0[Xx][0-9A-Fa-f]+|0[0-7]*)
decimal                 -?(([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([Ee][+-]?[0-9]+)?|[0-9]+[Ee][+-]?[0-9]+)
identifier              [_-]?[A-Za-z][0-9A-Z_a-z-]*
string                  \"[^\"]*\"
whitespace              [\t\n\r ]+
comment                 \/\/.*|\/\*(.|\n)*?\*\/
other                   [^\t\n\r 0-9A-Za-z]
literal                 ["\*\.\(\)\{\}\[\],;:=\+-\/~\|&\^\?<>]

%%

async        			return async;
attribute          		return attribute;
callback          		return callback;
const          			return CONST;
constructor          	return constructor;
deleter          		return deleter;
dictionary          	return dictionary;
enum          			return ENUM;
getter          		return getter;
includes          		return includes;
inherit          		return inherit;
interface          		return interface;
iterable          		return iterable;
maplike          		return maplike;
mixin          			return mixin;
namespace          		return NAMESPACE;
partial          		return partial;
readonly          		return readonly;
required          		return required;
setlike          		return setlike;
setter          		return setter;
static          		return STATIC;
stringifier          	return stringifier;
typedef          		return TYPEDEF;
unrestricted          	return unrestricted;
float                   return FLOAT;
double                  return DOUBLE;
true                    return TRUEK;
false                   return FALSEK;
unsigned                return UNSIGNED;
-Infinity               return NEGINF;
Infinity                return INF;
NaN                     return NaN;
null                    return null;
optional                return optional;
any                     return any;
other                   return other;
sequence 			    return sequence;
object       			return object;
symbol       			return symbol;
FrozenArray       		return FrozenArray;
ObservableArray       	return ObservableArray;
boolean       			return boolean;
byte       				return byte;
octet       			return octet;
bigint       			return bigint;
short       			return SHORT;
long       				return LONG;
Promise       			return Promise;
record       			return record;
ArrayBuffer       		return ArrayBuffer;
DataView       			return DataView;
Int8Array       		return Int8Array;
Int16Array       		return Int16Array;
Int32Array       		return Int32Array;
Uint8Array       		return Uint8Array;
Uint16Array       		return Uint16Array;
Uint32Array       		return Uint32Array;
Uint8ClampedArray       return Uint8ClampedArray;
Float32Array       		return Float32Array;
Float64Array       		return Float64Array;
undefined               return undefined;
ByteString              return ByteString;
DOMString               return DOMString;
USVString               return USVString;
or                      return OR;
{decimal}				return decimal;
{integer}				return integer;
{literal}				{
							return yytext[0];
						}
{identifier}			return identifier;
{string}			    return string;
\.\.\.                  return ellipsis;
{comment}
{whitespace}

%%

void webidlerror (const char *s)
{
	std::cerr << "Error: '" << s << "' at line " << webidllineno << std::endl;
}

namespace AeonGUI
{
	namespace WebIDL
	{
		int ParseString(const char* string)
		{
			YY_BUFFER_STATE state=webidl_scan_string(string);
			webidl_switch_to_buffer(state);
			BEGIN(INITIAL);
			return webidlparse();
		}
		int ParseFile(const char* filename)
		{
			int result = -1;
			yyin = fopen(filename, "r");
			if(yyin)
			{
				result = webidlparse();
				fclose(yyin);
			}
			return result;
		}
	}
}
