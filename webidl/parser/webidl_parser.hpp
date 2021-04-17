/* A Bison parser, made by GNU Bison 3.7.6.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_WEBIDL_C_CODE_AEONGUI_MINGW64_WEBIDL_PARSER_WEBIDL_PARSER_HPP_INCLUDED
# define YY_WEBIDL_C_CODE_AEONGUI_MINGW64_WEBIDL_PARSER_WEBIDL_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef WEBIDLDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define WEBIDLDEBUG 1
#  else
#   define WEBIDLDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define WEBIDLDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined WEBIDLDEBUG */
#if WEBIDLDEBUG
extern int webidldebug;
#endif
/* "%code requires" blocks.  */
#line 1 "C:/Code/AeonGUI/webidl/parser/webidl.ypp"

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

#line 72 "C:/Code/AeonGUI/mingw64/webidl/parser/webidl_parser.hpp"

/* Token kinds.  */
#ifndef WEBIDLTOKENTYPE
# define WEBIDLTOKENTYPE
enum webidltokentype
{
    WEBIDLEMPTY = -2,
    WEBIDLEOF = 0,                 /* "end of file"  */
    WEBIDLerror = 256,             /* error  */
    WEBIDLUNDEF = 257,             /* "invalid token"  */
    integer = 258,                 /* integer  */
    decimal = 259,                 /* decimal  */
    identifier = 260,              /* identifier  */
    string = 261,                  /* string  */
    null = 262,                    /* null  */
    ellipsis = 263,                /* ellipsis  */
    async = 264,                   /* async  */
    attribute = 265,               /* attribute  */
    callback = 266,                /* callback  */
    CONST = 267,                   /* CONST  */
    constructor = 268,             /* constructor  */
    deleter = 269,                 /* deleter  */
    dictionary = 270,              /* dictionary  */
    ENUM = 271,                    /* ENUM  */
    getter = 272,                  /* getter  */
    includes = 273,                /* includes  */
    inherit = 274,                 /* inherit  */
    interface = 275,               /* interface  */
        iterable = 276,                /* iterable  */
        maplike = 277,                 /* maplike  */
        mixin = 278,                   /* mixin  */
        NAMESPACE = 279,               /* NAMESPACE  */
        partial = 280,                 /* partial  */
        readonly = 281,                /* readonly  */
        required = 282,                /* required  */
        setlike = 283,                 /* setlike  */
        setter = 284,                  /* setter  */
        STATIC = 285,                  /* STATIC  */
        stringifier = 286,             /* stringifier  */
        TYPEDEF = 287,                 /* TYPEDEF  */
        unrestricted = 288,            /* unrestricted  */
        OR = 289,                      /* OR  */
        FLOAT = 290,                   /* FLOAT  */
        DOUBLE = 291,                  /* DOUBLE  */
        TRUEK = 292,                   /* TRUEK  */
        FALSEK = 293,                  /* FALSEK  */
        UNSIGNED = 294,                /* UNSIGNED  */
        INF = 295,                     /* INF  */
        NEGINF = 296,                  /* NEGINF  */
        NaN = 297,                     /* NaN  */
        optional = 298,                /* optional  */
        any = 299,                     /* any  */
        other = 300,                   /* other  */
        sequence = 301,                /* sequence  */
        object = 302,                  /* object  */
        symbol = 303,                  /* symbol  */
        FrozenArray = 304,             /* FrozenArray  */
        ObservableArray = 305,         /* ObservableArray  */
        boolean = 306,                 /* boolean  */
        byte = 307,                    /* byte  */
        octet = 308,                   /* octet  */
        bigint = 309,                  /* bigint  */
        SHORT = 310,                   /* SHORT  */
        LONG = 311,                    /* LONG  */
        Promise = 312,                 /* Promise  */
        record = 313,                  /* record  */
        ArrayBuffer = 314,             /* ArrayBuffer  */
        DataView = 315,                /* DataView  */
        Int8Array = 316,               /* Int8Array  */
        Int16Array = 317,              /* Int16Array  */
        Int32Array = 318,              /* Int32Array  */
        Uint8Array = 319,              /* Uint8Array  */
        Uint16Array = 320,             /* Uint16Array  */
        Uint32Array = 321,             /* Uint32Array  */
        Uint8ClampedArray = 322,       /* Uint8ClampedArray  */
        Float32Array = 323,            /* Float32Array  */
        Float64Array = 324,            /* Float64Array  */
        undefined = 325,               /* undefined  */
        ByteString = 326,              /* ByteString  */
        DOMString = 327,               /* DOMString  */
        USVString = 328                /* USVString  */
    };
typedef enum webidltokentype webidltoken_kind_t;
#endif

/* Value type.  */
#if ! defined WEBIDLSTYPE && ! defined WEBIDLSTYPE_IS_DECLARED
typedef int WEBIDLSTYPE;
# define WEBIDLSTYPE_IS_TRIVIAL 1
# define WEBIDLSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined WEBIDLLTYPE && ! defined WEBIDLLTYPE_IS_DECLARED
typedef struct WEBIDLLTYPE WEBIDLLTYPE;
struct WEBIDLLTYPE
{
    int first_line;
    int first_column;
    int last_line;
    int last_column;
};
# define WEBIDLLTYPE_IS_DECLARED 1
# define WEBIDLLTYPE_IS_TRIVIAL 1
#endif


extern WEBIDLSTYPE webidllval;
extern WEBIDLLTYPE webidllloc;
int webidlparse ( void );

#endif /* !YY_WEBIDL_C_CODE_AEONGUI_MINGW64_WEBIDL_PARSER_WEBIDL_PARSER_HPP_INCLUDED  */
