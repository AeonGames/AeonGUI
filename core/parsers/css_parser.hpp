/* A Bison parser, made by GNU Bison 3.4.1.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2019 Free Software Foundation,
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

#ifndef YY_CSS_C_CODE_AEONGUI_MINGW64_CORE_CSS_PARSER_HPP_INCLUDED
# define YY_CSS_C_CODE_AEONGUI_MINGW64_CORE_CSS_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef CSSDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define CSSDEBUG 1
#  else
#   define CSSDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define CSSDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined CSSDEBUG */
#if CSSDEBUG
extern int cssdebug;
#endif

/* Token type.  */
#ifndef CSSTOKENTYPE
# define CSSTOKENTYPE
enum csstokentype
{
    ANGLE = 258,
    ATKEYWORD = 259,
    BAD_STRING = 260,
    BAD_URI = 261,
    CDC = 262,
    CDO = 263,
    CHARSET_SYM = 264,
    DASHMATCH = 265,
    DELIM = 266,
    DIMENSION = 267,
    EMS = 268,
    EXS = 269,
    FREQ = 270,
    FUNCTION = 271,
    HASH = 272,
    IDENT = 273,
    IMPORTANT_SYM = 274,
    IMPORT_SYM = 275,
    INCLUDES = 276,
    LENGTH = 277,
    MEDIA_SYM = 278,
    NUMBER = 279,
    PAGE_SYM = 280,
    PERCENTAGE = 281,
    S = 282,
    STRING = 283,
    TIME = 284,
    UNICODE_RANGE = 285,
    URI = 286
};
#endif

/* Value type.  */
#if ! defined CSSSTYPE && ! defined CSSSTYPE_IS_DECLARED
typedef int CSSSTYPE;
# define CSSSTYPE_IS_TRIVIAL 1
# define CSSSTYPE_IS_DECLARED 1
#endif


extern CSSSTYPE csslval;

int cssparse ( void );

#endif /* !YY_CSS_C_CODE_AEONGUI_MINGW64_CORE_CSS_PARSER_HPP_INCLUDED  */
