/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

#ifndef YY_STYLE_C_CODE_AEONGUI_MINGW64_CORE_STYLE_PARSER_HPP_INCLUDED
# define YY_STYLE_C_CODE_AEONGUI_MINGW64_CORE_STYLE_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef STYLEDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define STYLEDEBUG 1
#  else
#   define STYLEDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define STYLEDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined STYLEDEBUG */
#if STYLEDEBUG
extern int styledebug;
#endif
/* "%code requires" blocks.  */
#line 1 "C:/Code/AeonGUI/core/parsers/style.ypp"

/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba
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

#line 71 "C:/Code/AeonGUI/mingw64/core/style_parser.hpp"

/* Token type.  */
#ifndef STYLETOKENTYPE
# define STYLETOKENTYPE
enum styletokentype
{
    IDENT = 258,
    COLOR = 259,
    NUMBER = 260
};
#endif

/* Value type.  */
#if ! defined STYLESTYPE && ! defined STYLESTYPE_IS_DECLARED
typedef int STYLESTYPE;
# define STYLESTYPE_IS_TRIVIAL 1
# define STYLESTYPE_IS_DECLARED 1
#endif


extern STYLESTYPE stylelval;

int styleparse ( AeonGUI::AttributeMap& aAttributeMap );

#endif /* !YY_STYLE_C_CODE_AEONGUI_MINGW64_CORE_STYLE_PARSER_HPP_INCLUDED  */
