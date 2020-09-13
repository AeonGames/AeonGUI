/* A Bison parser, made by GNU Bison 3.6.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.6.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         DSTYPE
/* Substitute the variable and function names.  */
#define yyparse         dparse
#define yylex           dlex
#define yyerror         derror
#define yydebug         ddebug
#define yynerrs         dnerrs
#define yylval          dlval
#define yychar          dchar

/* First part of user prologue.  */
#line 15 "C:/Code/AeonGUI/core/parsers/path_data.ypp"

#define YY_NO_UNISTD_H 1
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include "../core/parsers/dstype.h"
#include "aeongui/DrawType.h"
extern int dlex();
extern "C"
{
    int derror ( std::vector<AeonGUI::DrawType>& aPath, const char *s );
}
static void Merge ( dstype& aLeft, dstype& aRight )
{
    auto& left = std::get<std::vector<AeonGUI::DrawType>> ( aLeft );
    auto& right = std::get<std::vector<AeonGUI::DrawType>> ( aRight );
    left.reserve ( left.size() + right.size() );
    left.insert ( left.end(), right.begin(), right.end() );
}
static void AddCommandToPath ( std::vector<AeonGUI::DrawType>& aPath, const AeonGUI::DrawType& aCommand, std::vector<AeonGUI::DrawType>& aArguments )
{
    aPath.reserve ( aPath.size() + aArguments.size() + 1 );
    aPath.emplace_back ( aCommand );
    aPath.insert ( aPath.end(), aArguments.begin(), aArguments.end() );
}
static dstype GetArcArgs ( const dstype& aRadii, const dstype& aRotation, const dstype& aLarge, const dstype& aSweep, const dstype& aEnd )
{
    return std::vector<AeonGUI::DrawType>
    {
        std::get<std::vector<AeonGUI::DrawType>> ( aRadii ) [0],
                                              std::get<std::vector<AeonGUI::DrawType>> ( aRadii ) [1],
                                              std::get<AeonGUI::DrawType> ( aRotation ),
                                              std::get<double> ( std::get<AeonGUI::DrawType> ( aLarge ) ) ? true : false,
                                              std::get<double> ( std::get<AeonGUI::DrawType> ( aSweep ) ) ? true : false,
                                              std::get<std::vector<AeonGUI::DrawType>> ( aEnd ) [0],
                                              std::get<std::vector<AeonGUI::DrawType>> ( aEnd ) [1]
    };
}

#line 120 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_D_C_CODE_AEONGUI_MINGW64_CORE_PATH_DATA_PARSER_HPP_INCLUDED
# define YY_D_C_CODE_AEONGUI_MINGW64_CORE_PATH_DATA_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef DDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define DDEBUG 1
#  else
#   define DDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define DDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined DDEBUG */
#if DDEBUG
extern int ddebug;
#endif
/* "%code requires" blocks.  */
#line 1 "C:/Code/AeonGUI/core/parsers/path_data.ypp"

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

#line 178 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"

/* Token kinds.  */
#ifndef DTOKENTYPE
# define DTOKENTYPE
enum dtokentype
{
    DEMPTY = -2,
    DEOF = 0,                      /* "end of file"  */
    Derror = 256,                  /* error  */
    DUNDEF = 257,                  /* "invalid token"  */
    NUMBER = 258                   /* NUMBER  */
};
typedef enum dtokentype dtoken_kind_t;
#endif

/* Value type.  */
#if ! defined DSTYPE && ! defined DSTYPE_IS_DECLARED
typedef int DSTYPE;
# define DSTYPE_IS_TRIVIAL 1
# define DSTYPE_IS_DECLARED 1
#endif


extern DSTYPE dlval;

int dparse ( std::vector<AeonGUI::DrawType>& aPath );

#endif /* !YY_D_C_CODE_AEONGUI_MINGW64_CORE_PATH_DATA_PARSER_HPP_INCLUDED  */
/* Symbol kind.  */
enum yysymbol_kind_t
{
    YYSYMBOL_YYEMPTY = -2,
    YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
    YYSYMBOL_YYerror = 1,                    /* error  */
    YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
    YYSYMBOL_NUMBER = 3,                     /* NUMBER  */
    YYSYMBOL_4_M_ = 4,                       /* 'M'  */
    YYSYMBOL_5_m_ = 5,                       /* 'm'  */
    YYSYMBOL_6_Z_ = 6,                       /* 'Z'  */
    YYSYMBOL_7_z_ = 7,                       /* 'z'  */
    YYSYMBOL_8_L_ = 8,                       /* 'L'  */
    YYSYMBOL_9_l_ = 9,                       /* 'l'  */
    YYSYMBOL_10_H_ = 10,                     /* 'H'  */
    YYSYMBOL_11_h_ = 11,                     /* 'h'  */
    YYSYMBOL_12_V_ = 12,                     /* 'V'  */
    YYSYMBOL_13_v_ = 13,                     /* 'v'  */
    YYSYMBOL_14_C_ = 14,                     /* 'C'  */
    YYSYMBOL_15_c_ = 15,                     /* 'c'  */
    YYSYMBOL_16_S_ = 16,                     /* 'S'  */
    YYSYMBOL_17_s_ = 17,                     /* 's'  */
    YYSYMBOL_18_Q_ = 18,                     /* 'Q'  */
    YYSYMBOL_19_q_ = 19,                     /* 'q'  */
    YYSYMBOL_20_T_ = 20,                     /* 'T'  */
    YYSYMBOL_21_t_ = 21,                     /* 't'  */
    YYSYMBOL_22_A_ = 22,                     /* 'A'  */
    YYSYMBOL_23_a_ = 23,                     /* 'a'  */
    YYSYMBOL_YYACCEPT = 24,                  /* $accept  */
    YYSYMBOL_25_svg_path = 25,               /* svg-path  */
    YYSYMBOL_26_moveto_drawto_command_groups = 26, /* moveto-drawto-command-groups  */
    YYSYMBOL_27_moveto_drawto_command_group = 27, /* moveto-drawto-command-group  */
    YYSYMBOL_28_drawto_commands = 28,        /* drawto-commands  */
    YYSYMBOL_29_drawto_command = 29,         /* drawto-command  */
    YYSYMBOL_moveto = 30,                    /* moveto  */
    YYSYMBOL_closepath = 31,                 /* closepath  */
    YYSYMBOL_lineto = 32,                    /* lineto  */
    YYSYMBOL_33_horizontal_lineto = 33,      /* horizontal-lineto  */
    YYSYMBOL_34_vertical_lineto = 34,        /* vertical-lineto  */
    YYSYMBOL_curveto = 35,                   /* curveto  */
    YYSYMBOL_36_smooth_curveto = 36,         /* smooth-curveto  */
    YYSYMBOL_37_quadratic_bezier_curveto = 37, /* quadratic-bezier-curveto  */
    YYSYMBOL_38_smooth_quadratic_bezier_curveto = 38, /* smooth-quadratic-bezier-curveto  */
    YYSYMBOL_39_elliptical_arc = 39,         /* elliptical-arc  */
    YYSYMBOL_40_elliptical_arc_argument_sequence = 40, /* elliptical-arc-argument-sequence  */
    YYSYMBOL_41_elliptical_arc_argument = 41, /* elliptical-arc-argument  */
    YYSYMBOL_42_triple_coordinate_pair_argument_sequence = 42, /* triple-coordinate-pair-argument-sequence  */
    YYSYMBOL_43_triple_coordinate_pair_argument = 43, /* triple-coordinate-pair-argument  */
    YYSYMBOL_44_double_coordinate_pair_argument_sequence = 44, /* double-coordinate-pair-argument-sequence  */
    YYSYMBOL_45_double_coordinate_pair_argument = 45, /* double-coordinate-pair-argument  */
    YYSYMBOL_46_single_coordinate_pair_argument_sequence = 46, /* single-coordinate-pair-argument-sequence  */
    YYSYMBOL_47_coordinate_pair = 47,        /* coordinate-pair  */
    YYSYMBOL_48_coordinate_argument_sequence = 48, /* coordinate-argument-sequence  */
    YYSYMBOL_coordinate = 49                 /* coordinate  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
/* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
/* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
/* The OS might guarantee only one guard page at the bottom of the stack,
   and a page size can be as small as 4096 bytes.  So we cannot safely
   invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
   to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc ( YYSIZE_T ); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free ( void * ); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined DSTYPE_IS_TRIVIAL && DSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
    yy_state_t yyss_alloc;
    YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  12
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   117

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  24
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  26
/* YYNRULES -- Number of rules.  */
#define YYNRULES  53
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  80

#define YYMAXUTOK   258


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
    0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,    22,     2,    14,     2,     2,
    2,     2,    10,     2,     2,     2,     8,     4,     2,     2,
    2,    18,     2,    16,    20,     2,    12,     2,     2,     2,
    6,     2,     2,     2,     2,     2,     2,    23,     2,    15,
    2,     2,     2,     2,    11,     2,     2,     2,     9,     5,
    2,     2,     2,    19,     2,    17,    21,     2,    13,     2,
    2,     2,     7,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     1,     2,     3
};

#if DDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
    0,    59,    59,    60,    63,    64,    66,    67,    70,    71,
    73,    74,    75,    76,    77,    78,    79,    80,    81,    83,
    85,    88,    89,    92,    94,    97,    98,   101,   102,   105,
    106,   109,   110,   113,   114,   117,   118,   121,   123,   126,
    127,   134,   140,   141,   148,   160,   162,   169,   176,   178,
    185,   188,   193,   200
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if DDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name ( yysymbol_kind_t yysymbol ) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
    "\"end of file\"", "error", "\"invalid token\"", "NUMBER", "'M'", "'m'",
    "'Z'", "'z'", "'L'", "'l'", "'H'", "'h'", "'V'", "'v'", "'C'", "'c'",
    "'S'", "'s'", "'Q'", "'q'", "'T'", "'t'", "'A'", "'a'", "$accept",
    "svg-path", "moveto-drawto-command-groups",
    "moveto-drawto-command-group", "drawto-commands", "drawto-command",
    "moveto", "closepath", "lineto", "horizontal-lineto", "vertical-lineto",
    "curveto", "smooth-curveto", "quadratic-bezier-curveto",
    "smooth-quadratic-bezier-curveto", "elliptical-arc",
    "elliptical-arc-argument-sequence", "elliptical-arc-argument",
    "triple-coordinate-pair-argument-sequence",
    "triple-coordinate-pair-argument",
    "double-coordinate-pair-argument-sequence",
    "double-coordinate-pair-argument",
    "single-coordinate-pair-argument-sequence", "coordinate-pair",
    "coordinate-argument-sequence", "coordinate", YY_NULLPTR
};

static const char *
yysymbol_name ( yysymbol_kind_t yysymbol )
{
    return yytname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
    0,   256,   257,   258,    77,   109,    90,   122,    76,   108,
    72,   104,    86,   118,    67,    99,    83,   115,    81,   113,
    84,   116,    65,    97
};
#endif

#define YYPACT_NINF (-63)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int8 yypact[] =
{
    21,     1,     1,    14,    21,   -63,    94,   -63,     1,   -63,
    1,     1,   -63,   -63,   -63,   -63,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,    94,   -63,   -63,   -63,   -63,   -63,   -63,   -63,
    -63,   -63,   -63,   -63,   -63,     1,     1,     1,   -63,     1,
    1,     1,     1,   -63,     1,     1,     1,   -63,     1,     1,
    1,     1,     1,     1,     1,   -63,     1,     1,   -63,   -63,
    -63,     1,   -63,   -63,   -63,     1,   -63,     1,     1,   -63
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
    2,     0,     0,     0,     3,     4,     6,    53,    19,    48,
    0,    20,     1,     5,    21,    22,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     7,     8,    10,    11,    12,    13,    14,    15,
    16,    17,    18,    49,    50,    23,    24,    25,    51,    26,
    27,    28,    29,    42,     0,    30,    31,    45,     0,    32,
    33,    34,    35,    36,    37,    39,     0,    38,     9,    52,
    43,     0,    46,    47,    40,     0,    44,     0,     0,    41
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
    -63,   -63,   -63,    16,   -63,     2,   -63,   -63,   -63,   -63,
        -63,   -63,   -63,   -63,   -63,   -63,     0,   -62,    10,   -49,
        -8,   -48,     7,    -1,    18,    22
    };

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
    -1,     3,     4,     5,    32,    33,     6,    34,    35,    36,
        37,    38,    39,    40,    41,    42,    64,    65,    52,    53,
        56,    57,     8,    58,    47,    10
    };

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
    9,     9,    74,    70,     7,    74,    70,    43,    72,    11,
    43,    72,    72,    72,    12,     9,     9,    59,    60,    61,
    13,    54,    54,    45,    46,     1,     2,     9,     9,    66,
    66,    67,    44,    55,    68,    62,    63,    49,    50,    51,
    48,    48,    48,    48,    43,    43,     0,     0,     0,     0,
    0,    54,     0,    71,    54,     0,     0,    73,     0,     0,
    0,    43,    43,    66,     0,     0,    66,     0,     0,    69,
    76,    69,    69,    69,     0,     0,     0,    79,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,    75,     0,
    0,     0,     0,     0,     0,     0,     0,    77,     0,    78,
    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
    24,    25,    26,    27,    28,    29,    30,    31
};

static const yytype_int8 yycheck[] =
{
    1,     2,    64,    52,     3,    67,    55,     8,    56,     2,
    11,    59,    60,    61,     0,    16,    17,    25,    26,    27,
    4,    22,    23,    16,    17,     4,     5,    28,    29,    30,
    31,    31,    10,    23,    32,    28,    29,    19,    20,    21,
    18,    19,    20,    21,    45,    46,    -1,    -1,    -1,    -1,
    -1,    52,    -1,    54,    55,    -1,    -1,    58,    -1,    -1,
    -1,    62,    63,    64,    -1,    -1,    67,    -1,    -1,    47,
    71,    49,    50,    51,    -1,    -1,    -1,    78,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    75,    -1,    77,
    6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    17,    18,    19,    20,    21,    22,    23
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
    0,     4,     5,    25,    26,    27,    30,     3,    46,    47,
    49,    46,     0,    27,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
    22,    23,    28,    29,    31,    32,    33,    34,    35,    36,
    37,    38,    39,    47,    49,    46,    46,    48,    49,    48,
    48,    48,    42,    43,    47,    42,    44,    45,    47,    44,
    44,    44,    46,    46,    40,    41,    47,    40,    29,    49,
    43,    47,    45,    47,    41,    49,    47,    49,    49,    47
};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
    0,    24,    25,    25,    26,    26,    27,    27,    28,    28,
    29,    29,    29,    29,    29,    29,    29,    29,    29,    30,
    30,    31,    31,    32,    32,    33,    33,    34,    34,    35,
    35,    36,    36,    37,    37,    38,    38,    39,    39,    40,
    40,    41,    42,    42,    43,    44,    44,    45,    46,    46,
    47,    48,    48,    49
};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
    0,     2,     0,     1,     1,     2,     1,     2,     1,     2,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
    2,     1,     1,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     1,
    2,     5,     1,     2,     3,     1,     2,     2,     1,     2,
    2,     1,     2,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = DEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == DEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (aPath, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use Derror or DUNDEF. */
#define YYERRCODE DUNDEF


/* Enable debugging if requested.  */
#if DDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
# ifndef YY_LOCATION_PRINT
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, aPath); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print ( FILE *yyo,
                        yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    FILE *yyoutput = yyo;
    YYUSE ( yyoutput );
    YYUSE ( aPath );
    if ( !yyvaluep )
    {
        return;
    }
# ifdef YYPRINT
    if ( yykind < YYNTOKENS )
    {
        YYPRINT ( yyo, yytoknum[yykind], *yyvaluep );
    }
# endif
    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    YYUSE ( yykind );
    YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print ( FILE *yyo,
                  yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    YYFPRINTF ( yyo, "%s %s (",
                yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name ( yykind ) );

    yy_symbol_value_print ( yyo, yykind, yyvaluep, aPath );
    YYFPRINTF ( yyo, ")" );
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print ( yy_state_t *yybottom, yy_state_t *yytop )
{
    YYFPRINTF ( stderr, "Stack now" );
    for ( ; yybottom <= yytop; yybottom++ )
    {
        int yybot = *yybottom;
        YYFPRINTF ( stderr, " %d", yybot );
    }
    YYFPRINTF ( stderr, "\n" );
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print ( yy_state_t *yyssp, YYSTYPE *yyvsp,
                  int yyrule, std::vector<AeonGUI::DrawType>& aPath )
{
    int yylno = yyrline[yyrule];
    int yynrhs = yyr2[yyrule];
    int yyi;
    YYFPRINTF ( stderr, "Reducing stack by rule %d (line %d):\n",
                yyrule - 1, yylno );
    /* The symbols being reduced.  */
    for ( yyi = 0; yyi < yynrhs; yyi++ )
    {
        YYFPRINTF ( stderr, "   $%d = ", yyi + 1 );
        yy_symbol_print ( stderr,
                          YY_ACCESSING_SYMBOL ( +yyssp[yyi + 1 - yynrhs] ),
                          &yyvsp[ ( yyi + 1 ) - ( yynrhs )], aPath );
        YYFPRINTF ( stderr, "\n" );
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, aPath); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !DDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !DDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct ( const char *yymsg,
             yysymbol_kind_t yykind, YYSTYPE *yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    YYUSE ( yyvaluep );
    YYUSE ( aPath );
    if ( !yymsg )
    {
        yymsg = "Deleting";
    }
    YY_SYMBOL_PRINT ( yymsg, yykind, yyvaluep, yylocationp );

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    YYUSE ( yykind );
    YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse ( std::vector<AeonGUI::DrawType>& aPath )
{
    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize;

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    int yyn;
    /* The return value of yyparse.  */
    int yyresult;
    /* Lookahead token as an internal (translated) token number.  */
    yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
    /* The variables used to return semantic value and location from the
       action routines.  */
    YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

    /* The number of symbols on the RHS of the reduced rule.
       Keep to zero when no symbol should be popped.  */
    int yylen = 0;

    yynerrs = 0;
    yystate = 0;
    yyerrstatus = 0;

    yystacksize = YYINITDEPTH;
    yyssp = yyss = yyssa;
    yyvsp = yyvs = yyvsa;


    YYDPRINTF ( ( stderr, "Starting parse\n" ) );

    yychar = DEMPTY; /* Cause a token to be read.  */
    goto yysetstate;


    /*------------------------------------------------------------.
    | yynewstate -- push a new state, which is found in yystate.  |
    `------------------------------------------------------------*/
yynewstate:
    /* In all cases, when you get here, the value and location stacks
       have just been pushed.  So pushing a state here evens the stacks.  */
    yyssp++;


    /*--------------------------------------------------------------------.
    | yysetstate -- set current state (the top of the stack) to yystate.  |
    `--------------------------------------------------------------------*/
yysetstate:
    YYDPRINTF ( ( stderr, "Entering state %d\n", yystate ) );
    YY_ASSERT ( 0 <= yystate && yystate < YYNSTATES );
    YY_IGNORE_USELESS_CAST_BEGIN
    *yyssp = YY_CAST ( yy_state_t, yystate );
    YY_IGNORE_USELESS_CAST_END
    YY_STACK_PRINT ( yyss, yyssp );

    if ( yyss + yystacksize - 1 <= yyssp )
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
        goto yyexhaustedlab;
#else
    {
        /* Get the current used size of the three stacks, in elements.  */
        YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
        {
            /* Give user a chance to reallocate the stack.  Use copies of
               these so that the &'s don't force the real ones into
               memory.  */
            yy_state_t *yyss1 = yyss;
            YYSTYPE *yyvs1 = yyvs;

            /* Each stack pointer address is followed by the size of the
               data in use in that stack, in bytes.  This used to be a
               conditional around just the two extra args, but that might
               be undefined if yyoverflow is a macro.  */
            yyoverflow ( YY_ ( "memory exhausted" ),
                         &yyss1, yysize * YYSIZEOF ( *yyssp ),
                         &yyvs1, yysize * YYSIZEOF ( *yyvsp ),
                         &yystacksize );
            yyss = yyss1;
            yyvs = yyvs1;
        }
# else /* defined YYSTACK_RELOCATE */
        /* Extend the stack our own way.  */
        if ( YYMAXDEPTH <= yystacksize )
        {
            goto yyexhaustedlab;
        }
        yystacksize *= 2;
        if ( YYMAXDEPTH < yystacksize )
        {
            yystacksize = YYMAXDEPTH;
        }

        {
            yy_state_t *yyss1 = yyss;
            union yyalloc *yyptr =
                    YY_CAST ( union yyalloc *,
                              YYSTACK_ALLOC ( YY_CAST ( YYSIZE_T, YYSTACK_BYTES ( yystacksize ) ) ) );
            if ( ! yyptr )
            {
                goto yyexhaustedlab;
            }
            YYSTACK_RELOCATE ( yyss_alloc, yyss );
            YYSTACK_RELOCATE ( yyvs_alloc, yyvs );
#  undef YYSTACK_RELOCATE
            if ( yyss1 != yyssa )
            {
                YYSTACK_FREE ( yyss1 );
            }
        }
# endif

        yyssp = yyss + yysize - 1;
        yyvsp = yyvs + yysize - 1;

        YY_IGNORE_USELESS_CAST_BEGIN
        YYDPRINTF ( ( stderr, "Stack size increased to %ld\n",
                      YY_CAST ( long, yystacksize ) ) );
        YY_IGNORE_USELESS_CAST_END

        if ( yyss + yystacksize - 1 <= yyssp )
        {
            YYABORT;
        }
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

    if ( yystate == YYFINAL )
    {
        YYACCEPT;
    }

    goto yybackup;


    /*-----------.
    | yybackup.  |
    `-----------*/
yybackup:
    /* Do appropriate processing given the current state.  Read a
       lookahead token if we need one and don't already have one.  */

    /* First try to decide what to do without reference to lookahead token.  */
    yyn = yypact[yystate];
    if ( yypact_value_is_default ( yyn ) )
    {
        goto yydefault;
    }

    /* Not known => get a lookahead token if don't already have one.  */

    /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
    if ( yychar == DEMPTY )
    {
        YYDPRINTF ( ( stderr, "Reading a token\n" ) );
        yychar = yylex ();
    }

    if ( yychar <= DEOF )
    {
        yychar = DEOF;
        yytoken = YYSYMBOL_YYEOF;
        YYDPRINTF ( ( stderr, "Now at end of input.\n" ) );
    }
    else if ( yychar == Derror )
    {
        /* The scanner already issued an error message, process directly
           to error recovery.  But do not keep the error token as
           lookahead, it is too special and may lead us to an endless
           loop in error recovery. */
        yychar = DUNDEF;
        yytoken = YYSYMBOL_YYerror;
        goto yyerrlab1;
    }
    else
    {
        yytoken = YYTRANSLATE ( yychar );
        YY_SYMBOL_PRINT ( "Next token is", yytoken, &yylval, &yylloc );
    }

    /* If the proper action on seeing token YYTOKEN is to reduce or to
       detect an error, take that action.  */
    yyn += yytoken;
    if ( yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken )
    {
        goto yydefault;
    }
    yyn = yytable[yyn];
    if ( yyn <= 0 )
    {
        if ( yytable_value_is_error ( yyn ) )
        {
            goto yyerrlab;
        }
        yyn = -yyn;
        goto yyreduce;
    }

    /* Count tokens shifted since error; after three, turn off error
       status.  */
    if ( yyerrstatus )
    {
        yyerrstatus--;
    }

    /* Shift the lookahead token.  */
    YY_SYMBOL_PRINT ( "Shifting", yytoken, &yylval, &yylloc );
    yystate = yyn;
    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    *++yyvsp = yylval;
    YY_IGNORE_MAYBE_UNINITIALIZED_END

    /* Discard the shifted token.  */
    yychar = DEMPTY;
    goto yynewstate;


    /*-----------------------------------------------------------.
    | yydefault -- do the default action for the current state.  |
    `-----------------------------------------------------------*/
yydefault:
    yyn = yydefact[yystate];
    if ( yyn == 0 )
    {
        goto yyerrlab;
    }
    goto yyreduce;


    /*-----------------------------.
    | yyreduce -- do a reduction.  |
    `-----------------------------*/
yyreduce:
    /* yyn is the number of a rule to reduce with.  */
    yylen = yyr2[yyn];

    /* If YYLEN is nonzero, implement the default value of the action:
       '$$ = $1'.

       Otherwise, the following line sets YYVAL to garbage.
       This behavior is undocumented and Bison
       users should not rely upon it.  Assigning to YYVAL
       unconditionally makes the parser a bit smaller, and it avoids a
       GCC warning that YYVAL may be used uninitialized.  */
    yyval = yyvsp[1 - yylen];


    YY_REDUCE_PRINT ( yyn );
    switch ( yyn )
    {
    case 19:
#line 83 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1291 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 20:
#line 85 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1297 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 21:
#line 88 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            aPath.emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
        }
#line 1303 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 22:
#line 89 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            aPath.emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
        }
#line 1309 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 23:
#line 92 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1315 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 24:
#line 94 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1321 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 25:
#line 97 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1327 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 26:
#line 98 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1333 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 27:
#line 101 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1339 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 28:
#line 102 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1345 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 29:
#line 105 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1351 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 30:
#line 106 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1357 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 31:
#line 109 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1363 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 32:
#line 110 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1369 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 33:
#line 113 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1375 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 34:
#line 114 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1381 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 35:
#line 117 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1387 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 36:
#line 118 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1393 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 37:
#line 121 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1399 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 38:
#line 123 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1405 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 39:
#line 126 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1411 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 40:
#line 128 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1420 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 41:
#line 135 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-4], yyvsp[-3], yyvsp[-2], yyvsp[-1], yyvsp[0] );
        }
#line 1428 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 42:
#line 140 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1434 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 43:
#line 142 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1443 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 44:
#line 149 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            auto& left = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-2] );
            auto& center = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-1] );
            auto& right = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] );
            left.reserve ( left.size() + center.size() + right.size() );
            left.insert ( left.end(), center.begin(), center.end() );
            left.insert ( left.end(), right.begin(), right.end() );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1457 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 45:
#line 160 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1463 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 46:
#line 163 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1472 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 47:
#line 170 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1481 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 48:
#line 176 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1487 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 49:
#line 179 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1496 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 50:
#line 185 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::vector<AeonGUI::DrawType> {std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<AeonGUI::DrawType> ( yyvsp[0] ) };
        }
#line 1502 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 51:
#line 189 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::vector<AeonGUI::DrawType> {std::get<AeonGUI::DrawType> ( yyvsp[0] ) };
        }
#line 1510 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 52:
#line 194 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-1] ).emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1519 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;


#line 1523 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"

    default:
        break;
    }
    /* User semantic actions sometimes alter yychar, and that requires
       that yytoken be updated with the new translation.  We take the
       approach of translating immediately before every use of yytoken.
       One alternative is translating here after every semantic action,
       but that translation would be missed if the semantic action invokes
       YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
       if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
       incorrect destructor might then be invoked immediately.  In the
       case of YYERROR or YYBACKUP, subsequent parser actions might lead
       to an incorrect destructor call or verbose syntax error message
       before the lookahead is translated.  */
    YY_SYMBOL_PRINT ( "-> $$ =", YY_CAST ( yysymbol_kind_t, yyr1[yyn] ), &yyval, &yyloc );

    YYPOPSTACK ( yylen );
    yylen = 0;

    *++yyvsp = yyval;

    /* Now 'shift' the result of the reduction.  Determine what state
       that goes to, based on the state we popped back to and the rule
       number reduced by.  */
    {
        const int yylhs = yyr1[yyn] - YYNTOKENS;
        const int yyi = yypgoto[yylhs] + *yyssp;
        yystate = ( 0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
                    ? yytable[yyi]
                    : yydefgoto[yylhs] );
    }

    goto yynewstate;


    /*--------------------------------------.
    | yyerrlab -- here on detecting error.  |
    `--------------------------------------*/
yyerrlab:
    /* Make sure we have latest lookahead translation.  See comments at
       user semantic actions for why this is necessary.  */
    yytoken = yychar == DEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE ( yychar );
    /* If not already recovering from an error, report this error.  */
    if ( !yyerrstatus )
    {
        ++yynerrs;
        yyerror ( aPath, YY_ ( "syntax error" ) );
    }

    if ( yyerrstatus == 3 )
    {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        if ( yychar <= DEOF )
        {
            /* Return failure if at end of input.  */
            if ( yychar == DEOF )
            {
                YYABORT;
            }
        }
        else
        {
            yydestruct ( "Error: discarding",
                         yytoken, &yylval, aPath );
            yychar = DEMPTY;
        }
    }

    /* Else will try to reuse lookahead token after shifting the error
       token.  */
    goto yyerrlab1;


    /*---------------------------------------------------.
    | yyerrorlab -- error raised explicitly by YYERROR.  |
    `---------------------------------------------------*/
yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and the
       label yyerrorlab therefore never appears in user code.  */
    if ( 0 )
    {
        YYERROR;
    }

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    YYPOPSTACK ( yylen );
    yylen = 0;
    YY_STACK_PRINT ( yyss, yyssp );
    yystate = *yyssp;
    goto yyerrlab1;


    /*-------------------------------------------------------------.
    | yyerrlab1 -- common code for both syntax error and YYERROR.  |
    `-------------------------------------------------------------*/
yyerrlab1:
    yyerrstatus = 3;      /* Each real token shifted decrements this.  */

    /* Pop stack until we find a state that shifts the error token.  */
    for ( ;; )
    {
        yyn = yypact[yystate];
        if ( !yypact_value_is_default ( yyn ) )
        {
            yyn += YYSYMBOL_YYerror;
            if ( 0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror )
            {
                yyn = yytable[yyn];
                if ( 0 < yyn )
                {
                    break;
                }
            }
        }

        /* Pop the current state because it cannot handle the error token.  */
        if ( yyssp == yyss )
        {
            YYABORT;
        }


        yydestruct ( "Error: popping",
                     YY_ACCESSING_SYMBOL ( yystate ), yyvsp, aPath );
        YYPOPSTACK ( 1 );
        yystate = *yyssp;
        YY_STACK_PRINT ( yyss, yyssp );
    }

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    *++yyvsp = yylval;
    YY_IGNORE_MAYBE_UNINITIALIZED_END


    /* Shift the error token.  */
    YY_SYMBOL_PRINT ( "Shifting", YY_ACCESSING_SYMBOL ( yyn ), yyvsp, yylsp );

    yystate = yyn;
    goto yynewstate;


    /*-------------------------------------.
    | yyacceptlab -- YYACCEPT comes here.  |
    `-------------------------------------*/
yyacceptlab:
    yyresult = 0;
    goto yyreturn;


    /*-----------------------------------.
    | yyabortlab -- YYABORT comes here.  |
    `-----------------------------------*/
yyabortlab:
    yyresult = 1;
    goto yyreturn;


#if !defined yyoverflow
    /*-------------------------------------------------.
    | yyexhaustedlab -- memory exhaustion comes here.  |
    `-------------------------------------------------*/
yyexhaustedlab:
    yyerror ( aPath, YY_ ( "memory exhausted" ) );
    yyresult = 2;
    /* Fall through.  */
#endif


    /*-----------------------------------------------------.
    | yyreturn -- parsing is finished, return the result.  |
    `-----------------------------------------------------*/
yyreturn:
    if ( yychar != DEMPTY )
    {
        /* Make sure we have latest lookahead translation.  See comments at
           user semantic actions for why this is necessary.  */
        yytoken = YYTRANSLATE ( yychar );
        yydestruct ( "Cleanup: discarding lookahead",
                     yytoken, &yylval, aPath );
    }
    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    YYPOPSTACK ( yylen );
    YY_STACK_PRINT ( yyss, yyssp );
    while ( yyssp != yyss )
    {
        yydestruct ( "Cleanup: popping",
                     YY_ACCESSING_SYMBOL ( +*yyssp ), yyvsp, aPath );
        YYPOPSTACK ( 1 );
    }
#ifndef yyoverflow
    if ( yyss != yyssa )
    {
        YYSTACK_FREE ( yyss );
    }
#endif

    return yyresult;
}

#line 201 "C:/Code/AeonGUI/core/parsers/path_data.ypp"

extern "C"
{
    int derror ( std::vector<AeonGUI::DrawType>& aPath, const char *s )
    {
        std::cerr << s << std::endl;
        return 0;
    }
}
