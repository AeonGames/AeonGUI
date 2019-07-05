/* A Bison parser, made by GNU Bison 3.4.1.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.4.1"

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
#line 1 "C:/Code/AeonGUI/core/parsers/path_data.ypp"

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
                                              std::get<AeonGUI::DrawType> ( aLarge ),
                                              std::get<AeonGUI::DrawType> ( aSweep ),
                                              std::get<std::vector<AeonGUI::DrawType>> ( aEnd ) [0],
                                              std::get<std::vector<AeonGUI::DrawType>> ( aEnd ) [1]
    };
}

#line 131 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"

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

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

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

/* Token type.  */
#ifndef DTOKENTYPE
# define DTOKENTYPE
enum dtokentype
{
    FLAG = 258,
    NUMBER = 259
};
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



#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

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

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
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


#define YY_ASSERT(E) ((void) (0 && (E)))

#if ! defined yyoverflow || YYERROR_VERBOSE

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
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined DSTYPE_IS_TRIVIAL && DSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
    yytype_int16 yyss_alloc;
    YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
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
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  13
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   222

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  26
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  26
/* YYNRULES -- Number of rules.  */
#define YYNRULES  77
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  140

#define YYUNDEFTOK  2
#define YYMAXUTOK   259

/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                                \
  ((unsigned) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_uint8 yytranslate[] =
{
    0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,    25,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,    23,     2,    15,     2,     2,
    2,     2,    11,     2,     2,     2,     9,     5,     2,     2,
    2,    19,     2,    17,    21,     2,    13,     2,     2,     2,
    7,     2,     2,     2,     2,     2,     2,    24,     2,    16,
    2,     2,     2,     2,    12,     2,     2,     2,    10,     6,
    2,     2,     2,    20,     2,    18,    22,     2,    14,     2,
    2,     2,     8,     2,     2,     2,     2,     2,     2,     2,
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
    2,     2,     2,     2,     2,     2,     1,     2,     3,     4
};

#if DDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
    0,    56,    56,    57,    60,    61,    63,    64,    67,    68,
    70,    71,    72,    73,    74,    75,    76,    77,    78,    80,
    82,    85,    86,    89,    91,    94,    95,    98,    99,   102,
    103,   106,   107,   110,   111,   114,   115,   118,   120,   123,
    124,   129,   136,   141,   146,   151,   156,   161,   166,   171,
    176,   181,   186,   191,   196,   201,   206,   211,   217,   218,
    223,   230,   241,   253,   255,   260,   267,   273,   280,   282,
    288,   295,   297,   300,   305,   311,   318,   320
};
#endif

#if DDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
    "$end", "error", "$undefined", "FLAG", "NUMBER", "'M'", "'m'", "'Z'",
    "'z'", "'L'", "'l'", "'H'", "'h'", "'V'", "'v'", "'C'", "'c'", "'S'",
    "'s'", "'Q'", "'q'", "'T'", "'t'", "'A'", "'a'", "','", "$accept",
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
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
    0,   256,   257,   258,   259,    77,   109,    90,   122,    76,
    108,    72,   104,    86,   118,    67,    99,    83,   115,    81,
    113,    84,   116,    65,    97,    44
};
# endif

#define YYPACT_NINF -61

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-61)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    140,   155,   155,     2,   140,   -61,   198,   -61,   -61,     0,
    -61,     9,     0,   -61,   -61,   -61,   -61,   155,   155,   155,
    155,   155,   155,   155,   155,   155,   155,   155,   155,   155,
    155,   155,   155,   198,   -61,   -61,   -61,   -61,   -61,   -61,
    -61,   -61,   -61,   -61,   155,   -61,   155,   -61,     0,     0,
    11,   -61,    11,    11,    11,    15,   -61,    34,    15,    38,
    -61,    42,    38,    38,    38,     0,     0,    46,   -61,    48,
    46,   -61,   -61,   -61,   155,   -61,   155,   -61,   155,   155,
    155,   -61,   155,   -61,   155,   -61,   155,    36,   -61,   -61,
    1,   -61,   -61,   -61,   -61,    71,    73,    32,   155,    96,
    41,    76,    50,   100,   -61,    81,    52,   102,   155,   -61,
    83,    86,    55,   155,   -61,    88,    91,    59,   -61,   155,
    -61,   155,   -61,   129,   -61,   155,   -61,   155,   -61,   138,
    -61,   -61,   155,   -61,   -61,   -61,   155,   -61,   -61,   -61
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
    2,     0,     0,     0,     3,     4,     6,    77,    76,    19,
    68,     0,    20,     1,     5,    21,    22,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     7,     8,    10,    11,    12,    13,    14,
    15,    16,    17,    18,     0,    69,     0,    71,    23,    24,
    25,    73,    26,    27,    28,    29,    58,     0,    30,    31,
    63,     0,    32,    33,    34,    35,    36,    37,    39,     0,
    38,     9,    70,    72,     0,    74,     0,    59,     0,     0,
    0,    64,     0,    66,     0,    40,     0,     0,    75,    60,
    0,    61,    65,    67,    41,     0,     0,     0,     0,     0,
    0,     0,     0,     0,    62,     0,     0,     0,     0,    42,
    0,     0,     0,     0,    43,     0,     0,     0,    46,     0,
    45,     0,    44,     0,    49,     0,    51,     0,    47,     0,
    48,    52,     0,    50,    54,    55,     0,    56,    53,    57
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -61,   -61,   -61,    64,   -61,    37,   -61,   -61,   -61,   -61,
        -61,   -61,   -61,   -61,   -61,   -61,    40,   -60,    58,   -49,
        124,    85,     3,    -1,   135,   118
    };

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
    -1,     3,     4,     5,    33,    34,     6,    35,    36,    37,
        38,    39,    40,    41,    42,    43,    67,    68,    55,    56,
        59,    60,     9,    61,    50,    11
    };

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
    10,    10,    13,     7,     8,    12,    77,    85,    45,    77,
    85,    45,     7,     8,     7,     8,    10,    10,     7,     8,
    48,    49,    57,    57,    94,    44,    98,    89,    10,    10,
    69,    69,    65,    66,    46,   103,    74,     7,     8,    96,
    76,     7,     8,    72,   107,     7,     8,    45,    45,     7,
    8,     7,     8,   110,    57,   115,    79,    57,   123,    78,
    83,    97,   129,    80,    45,    45,    69,    82,    14,    69,
    71,    84,    70,    86,    99,    57,   101,    90,    91,     7,
    8,    93,    58,    69,     7,     8,     7,     8,     0,     7,
    8,     7,     8,     0,     7,     8,   100,   104,   102,   105,
    109,   108,     0,   111,   114,   116,   113,   118,   119,   120,
    122,   121,   124,   125,   126,   128,   127,     0,   130,     0,
    131,   106,   133,     0,   134,   112,   135,   117,   137,    47,
    0,   138,     7,     8,     0,   139,     0,    51,    51,    51,
    51,     7,     8,     0,    81,     1,     2,    81,    81,    81,
    62,    63,    64,     0,   132,    52,    53,    54,     7,     8,
    0,     0,     0,   136,    73,    92,     0,     0,    75,     0,
    75,    75,    75,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,    87,     0,     0,
    0,     0,    88,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,    95,    15,    16,    17,    18,    19,
    20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
    30,    31,    32
};

static const yytype_int16 yycheck[] =
{
    1,     2,     0,     3,     4,     2,    55,    67,     9,    58,
    70,    12,     3,     4,     3,     4,    17,    18,     3,     4,
    17,    18,    23,    24,    84,    25,    25,    76,    29,    30,
    31,    32,    29,    30,    25,     3,    25,     3,     4,     3,
    25,     3,     4,    44,     3,     3,     4,    48,    49,     3,
    4,     3,     4,     3,    55,     3,    57,    58,     3,    25,
    61,    25,     3,    25,    65,    66,    67,    25,     4,    70,
    33,    25,    32,    25,     3,    76,     3,    78,    79,     3,
    4,    82,    24,    84,     3,     4,     3,     4,    -1,     3,
    4,     3,     4,    -1,     3,     4,    25,    98,    25,     3,
    101,    25,    -1,     3,   105,     3,    25,   108,    25,   110,
    111,    25,   113,    25,   115,   116,    25,    -1,   119,    -1,
    121,    25,   123,    -1,   125,    25,   127,    25,   129,    11,
    -1,   132,     3,     4,    -1,   136,    -1,    19,    20,    21,
    22,     3,     4,    -1,    59,     5,     6,    62,    63,    64,
    26,    27,    28,    -1,    25,    20,    21,    22,     3,     4,
    -1,    -1,    -1,    25,    46,    80,    -1,    -1,    50,    -1,
    52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    69,    -1,    -1,
    -1,    -1,    74,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    86,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
    22,    23,    24
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
    0,     5,     6,    27,    28,    29,    32,     3,     4,    48,
    49,    51,    48,     0,    29,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
    22,    23,    24,    30,    31,    33,    34,    35,    36,    37,
    38,    39,    40,    41,    25,    49,    25,    51,    48,    48,
    50,    51,    50,    50,    50,    44,    45,    49,    44,    46,
    47,    49,    46,    46,    46,    48,    48,    42,    43,    49,
    42,    31,    49,    51,    25,    51,    25,    45,    25,    49,
    25,    47,    25,    49,    25,    43,    25,    51,    51,    45,
    49,    49,    47,    49,    43,    51,     3,    25,    25,     3,
    25,     3,    25,     3,    49,     3,    25,     3,    25,    49,
    3,     3,    25,    25,    49,     3,     3,    25,    49,    25,
    49,    25,    49,     3,    49,    25,    49,    25,    49,     3,
    49,    49,    25,    49,    49,    49,    25,    49,    49,    49
};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
    0,    26,    27,    27,    28,    28,    29,    29,    30,    30,
    31,    31,    31,    31,    31,    31,    31,    31,    31,    32,
    32,    33,    33,    34,    34,    35,    35,    36,    36,    37,
    37,    38,    38,    39,    39,    40,    40,    41,    41,    42,
    42,    42,    43,    43,    43,    43,    43,    43,    43,    43,
    43,    43,    43,    43,    43,    43,    43,    43,    44,    44,
    44,    45,    45,    46,    46,    46,    47,    47,    48,    48,
    48,    49,    49,    50,    50,    50,    51,    51
};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
    0,     2,     0,     1,     1,     2,     1,     2,     1,     2,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
    2,     1,     1,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     1,
    2,     3,     5,     6,     6,     6,     6,     7,     7,     7,
    7,     7,     7,     8,     8,     8,     8,     9,     1,     2,
    3,     3,     5,     1,     2,     3,     2,     3,     1,     2,
    3,     2,     3,     1,     2,     3,     1,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
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

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



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
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, aPath); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print ( FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    FILE *yyoutput = yyo;
    YYUSE ( yyoutput );
    YYUSE ( aPath );
    if ( !yyvaluep )
    {
        return;
    }
# ifdef YYPRINT
    if ( yytype < YYNTOKENS )
    {
        YYPRINT ( yyo, yytoknum[yytype], *yyvaluep );
    }
# endif
    YYUSE ( yytype );
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print ( FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    YYFPRINTF ( yyo, "%s %s (",
                yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype] );

    yy_symbol_value_print ( yyo, yytype, yyvaluep, aPath );
    YYFPRINTF ( yyo, ")" );
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print ( yytype_int16 *yybottom, yytype_int16 *yytop )
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
yy_reduce_print ( yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule, std::vector<AeonGUI::DrawType>& aPath )
{
    unsigned long yylno = yyrline[yyrule];
    int yynrhs = yyr2[yyrule];
    int yyi;
    YYFPRINTF ( stderr, "Reducing stack by rule %d (line %lu):\n",
                yyrule - 1, yylno );
    /* The symbols being reduced.  */
    for ( yyi = 0; yyi < yynrhs; yyi++ )
    {
        YYFPRINTF ( stderr, "   $%d = ", yyi + 1 );
        yy_symbol_print ( stderr,
                          yystos[yyssp[yyi + 1 - yynrhs]],
                          &yyvsp[ ( yyi + 1 ) - ( yynrhs )]
                          , aPath );
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
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
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


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen ( const char *yystr )
{
    YYSIZE_T yylen;
    for ( yylen = 0; yystr[yylen]; yylen++ )
    {
        continue;
    }
    return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy ( char *yydest, const char *yysrc )
{
    char *yyd = yydest;
    const char *yys = yysrc;

    while ( ( *yyd++ = *yys++ ) != '\0' )
    {
        continue;
    }

    return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr ( char *yyres, const char *yystr )
{
    if ( *yystr == '"' )
    {
        YYSIZE_T yyn = 0;
        char const *yyp = yystr;

        for ( ;; )
            switch ( *++yyp )
            {
            case '\'':
            case ',':
                goto do_not_strip_quotes;

            case '\\':
                if ( *++yyp != '\\' )
                {
                    goto do_not_strip_quotes;
                }
                else
                {
                    goto append;
                }

append:
            default:
                if ( yyres )
                {
                    yyres[yyn] = *yyp;
                }
                yyn++;
                break;

            case '"':
                if ( yyres )
                {
                    yyres[yyn] = '\0';
                }
                return yyn;
            }
do_not_strip_quotes:
        ;
    }

    if ( ! yyres )
    {
        return yystrlen ( yystr );
    }

    return ( YYSIZE_T ) ( yystpcpy ( yyres, yystr ) - yyres );
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error ( YYSIZE_T *yymsg_alloc, char **yymsg,
                 yytype_int16 *yyssp, int yytoken )
{
    YYSIZE_T yysize0 = yytnamerr ( YY_NULLPTR, yytname[yytoken] );
    YYSIZE_T yysize = yysize0;
    enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
    /* Internationalized format string. */
    const char *yyformat = YY_NULLPTR;
    /* Arguments of yyformat. */
    char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
    /* Number of reported tokens (one for the "unexpected", one per
       "expected"). */
    int yycount = 0;

    /* There are many possibilities here to consider:
       - If this state is a consistent state with a default action, then
         the only way this function was invoked is if the default action
         is an error action.  In that case, don't check for expected
         tokens because there are none.
       - The only way there can be no lookahead present (in yychar) is if
         this state is a consistent state with a default action.  Thus,
         detecting the absence of a lookahead is sufficient to determine
         that there is no unexpected or expected token to report.  In that
         case, just report a simple "syntax error".
       - Don't assume there isn't a lookahead just because this state is a
         consistent state with a default action.  There might have been a
         previous inconsistent state, consistent state with a non-default
         action, or user semantic action that manipulated yychar.
       - Of course, the expected token list depends on states to have
         correct lookahead information, and it depends on the parser not
         to perform extra reductions after fetching a lookahead from the
         scanner and before detecting a syntax error.  Thus, state merging
         (from LALR or IELR) and default reductions corrupt the expected
         token list.  However, the list is correct for canonical LR with
         one exception: it will still contain any token that will not be
         accepted due to an error action in a later state.
    */
    if ( yytoken != YYEMPTY )
    {
        int yyn = yypact[*yyssp];
        yyarg[yycount++] = yytname[yytoken];
        if ( !yypact_value_is_default ( yyn ) )
        {
            /* Start YYX at -YYN if negative to avoid negative indexes in
               YYCHECK.  In other words, skip the first -YYN actions for
               this state because they are default actions.  */
            int yyxbegin = yyn < 0 ? -yyn : 0;
            /* Stay within bounds of both yycheck and yytname.  */
            int yychecklim = YYLAST - yyn + 1;
            int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
            int yyx;

            for ( yyx = yyxbegin; yyx < yyxend; ++yyx )
                if ( yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                     && !yytable_value_is_error ( yytable[yyx + yyn] ) )
                {
                    if ( yycount == YYERROR_VERBOSE_ARGS_MAXIMUM )
                    {
                        yycount = 1;
                        yysize = yysize0;
                        break;
                    }
                    yyarg[yycount++] = yytname[yyx];
                    {
                        YYSIZE_T yysize1 = yysize + yytnamerr ( YY_NULLPTR, yytname[yyx] );
                        if ( yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM )
                        {
                            yysize = yysize1;
                        }
                        else
                        {
                            return 2;
                        }
                    }
                }
        }
    }

    switch ( yycount )
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
        YYCASE_ ( 0, YY_ ( "syntax error" ) );
        YYCASE_ ( 1, YY_ ( "syntax error, unexpected %s" ) );
        YYCASE_ ( 2, YY_ ( "syntax error, unexpected %s, expecting %s" ) );
        YYCASE_ ( 3, YY_ ( "syntax error, unexpected %s, expecting %s or %s" ) );
        YYCASE_ ( 4, YY_ ( "syntax error, unexpected %s, expecting %s or %s or %s" ) );
        YYCASE_ ( 5, YY_ ( "syntax error, unexpected %s, expecting %s or %s or %s or %s" ) );
# undef YYCASE_
    }

    {
        YYSIZE_T yysize1 = yysize + yystrlen ( yyformat );
        if ( yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM )
        {
            yysize = yysize1;
        }
        else
        {
            return 2;
        }
    }

    if ( *yymsg_alloc < yysize )
    {
        *yymsg_alloc = 2 * yysize;
        if ( ! ( yysize <= *yymsg_alloc
                 && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM ) )
        {
            *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
        }
        return 1;
    }

    /* Avoid sprintf, as that infringes on the user's name space.
       Don't have undefined behavior even if the translation
       produced a string with the wrong number of "%s"s.  */
    {
        char *yyp = *yymsg;
        int yyi = 0;
        while ( ( *yyp = *yyformat ) != '\0' )
            if ( *yyp == '%' && yyformat[1] == 's' && yyi < yycount )
            {
                yyp += yytnamerr ( yyp, yyarg[yyi++] );
                yyformat += 2;
            }
            else
            {
                yyp++;
                yyformat++;
            }
    }
    return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct ( const char *yymsg, int yytype, YYSTYPE *yyvaluep, std::vector<AeonGUI::DrawType>& aPath )
{
    YYUSE ( yyvaluep );
    YYUSE ( aPath );
    if ( !yymsg )
    {
        yymsg = "Deleting";
    }
    YY_SYMBOL_PRINT ( yymsg, yytype, yyvaluep, yylocationp );

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    YYUSE ( yytype );
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
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

    int yyn;
    int yyresult;
    /* Lookahead token as an internal (translated) token number.  */
    int yytoken = 0;
    /* The variables used to return semantic value and location from the
       action routines.  */
    YYSTYPE yyval;

#if YYERROR_VERBOSE
    /* Buffer for error messages, and its allocated size.  */
    char yymsgbuf[128];
    char *yymsg = yymsgbuf;
    YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

    /* The number of symbols on the RHS of the reduced rule.
       Keep to zero when no symbol should be popped.  */
    int yylen = 0;

    yyssp = yyss = yyssa;
    yyvsp = yyvs = yyvsa;
    yystacksize = YYINITDEPTH;

    YYDPRINTF ( ( stderr, "Starting parse\n" ) );

    yystate = 0;
    yyerrstatus = 0;
    yynerrs = 0;
    yychar = YYEMPTY; /* Cause a token to be read.  */
    goto yysetstate;


    /*------------------------------------------------------------.
    | yynewstate -- push a new state, which is found in yystate.  |
    `------------------------------------------------------------*/
yynewstate:
    /* In all cases, when you get here, the value and location stacks
       have just been pushed.  So pushing a state here evens the stacks.  */
    yyssp++;


    /*--------------------------------------------------------------------.
    | yynewstate -- set current state (the top of the stack) to yystate.  |
    `--------------------------------------------------------------------*/
yysetstate:
    YYDPRINTF ( ( stderr, "Entering state %d\n", yystate ) );
    YY_ASSERT ( 0 <= yystate && yystate < YYNSTATES );
    *yyssp = ( yytype_int16 ) yystate;

    if ( yyss + yystacksize - 1 <= yyssp )
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
        goto yyexhaustedlab;
#else
    {
        /* Get the current used size of the three stacks, in elements.  */
        YYSIZE_T yysize = ( YYSIZE_T ) ( yyssp - yyss + 1 );

# if defined yyoverflow
        {
            /* Give user a chance to reallocate the stack.  Use copies of
               these so that the &'s don't force the real ones into
               memory.  */
            YYSTYPE *yyvs1 = yyvs;
            yytype_int16 *yyss1 = yyss;

            /* Each stack pointer address is followed by the size of the
               data in use in that stack, in bytes.  This used to be a
               conditional around just the two extra args, but that might
               be undefined if yyoverflow is a macro.  */
            yyoverflow ( YY_ ( "memory exhausted" ),
                         &yyss1, yysize * sizeof ( *yyssp ),
                         &yyvs1, yysize * sizeof ( *yyvsp ),
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
            yytype_int16 *yyss1 = yyss;
            union yyalloc *yyptr =
                    ( union yyalloc * ) YYSTACK_ALLOC ( YYSTACK_BYTES ( yystacksize ) );
            if ( ! yyptr )
            {
                goto yyexhaustedlab;
            }
            YYSTACK_RELOCATE ( yyss_alloc, yyss );
            YYSTACK_RELOCATE ( yyvs_alloc, yyvs );
# undef YYSTACK_RELOCATE
            if ( yyss1 != yyssa )
            {
                YYSTACK_FREE ( yyss1 );
            }
        }
# endif

        yyssp = yyss + yysize - 1;
        yyvsp = yyvs + yysize - 1;

        YYDPRINTF ( ( stderr, "Stack size increased to %lu\n",
                      ( unsigned long ) yystacksize ) );

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

    /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
    if ( yychar == YYEMPTY )
    {
        YYDPRINTF ( ( stderr, "Reading a token: " ) );
        yychar = yylex ();
    }

    if ( yychar <= YYEOF )
    {
        yychar = yytoken = YYEOF;
        YYDPRINTF ( ( stderr, "Now at end of input.\n" ) );
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

    /* Discard the shifted token.  */
    yychar = YYEMPTY;

    yystate = yyn;
    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    *++yyvsp = yylval;
    YY_IGNORE_MAYBE_UNINITIALIZED_END
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
#line 80 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1388 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 20:
#line 82 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1394 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 21:
#line 85 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            aPath.emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
        }
#line 1400 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 22:
#line 86 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            aPath.emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
        }
#line 1406 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 23:
#line 89 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1412 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 24:
#line 91 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1418 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 25:
#line 94 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1424 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 26:
#line 95 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1430 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 27:
#line 98 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1436 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 28:
#line 99 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1442 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 29:
#line 102 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1448 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 30:
#line 103 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1454 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 31:
#line 106 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1460 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 32:
#line 107 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1466 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 33:
#line 110 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1472 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 34:
#line 111 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1478 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 35:
#line 114 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1484 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 36:
#line 115 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1490 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 37:
#line 118 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1496 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 38:
#line 120 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            AddCommandToPath ( aPath, std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] ) );
        }
#line 1502 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 39:
#line 123 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1508 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 40:
#line 125 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1517 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 41:
#line 130 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-2], yyvsp[-1] );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1526 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 42:
#line 137 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-4], yyvsp[-3], yyvsp[-2], yyvsp[-1], yyvsp[0] );
        }
#line 1534 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 43:
#line 142 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-5], yyvsp[-3], yyvsp[-2], yyvsp[-1], yyvsp[0] );
        }
#line 1542 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 44:
#line 147 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-5], yyvsp[-4], yyvsp[-2], yyvsp[-1], yyvsp[0] );
        }
#line 1550 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 45:
#line 152 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-5], yyvsp[-4], yyvsp[-3], yyvsp[-1], yyvsp[0] );
        }
#line 1558 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 46:
#line 157 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-5], yyvsp[-4], yyvsp[-3], yyvsp[-2], yyvsp[0] );
        }
#line 1566 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 47:
#line 162 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-4], yyvsp[-2], yyvsp[-1], yyvsp[0] );
        }
#line 1574 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 48:
#line 167 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-5], yyvsp[-4], yyvsp[-2], yyvsp[0] );
        }
#line 1582 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 49:
#line 172 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-4], yyvsp[-3], yyvsp[-2], yyvsp[0] );
        }
#line 1590 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 50:
#line 177 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-5], yyvsp[-3], yyvsp[-1], yyvsp[0] );
        }
#line 1598 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 51:
#line 182 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-4], yyvsp[-3], yyvsp[-1], yyvsp[0] );
        }
#line 1606 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 52:
#line 187 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-6], yyvsp[-5], yyvsp[-3], yyvsp[-2], yyvsp[0] );
        }
#line 1614 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 53:
#line 192 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-7], yyvsp[-6], yyvsp[-4], yyvsp[-2], yyvsp[0] );
        }
#line 1622 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 54:
#line 197 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-7], yyvsp[-5], yyvsp[-4], yyvsp[-2], yyvsp[0] );
        }
#line 1630 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 55:
#line 202 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-7], yyvsp[-5], yyvsp[-3], yyvsp[-2], yyvsp[0] );
        }
#line 1638 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 56:
#line 207 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-7], yyvsp[-5], yyvsp[-3], yyvsp[-1], yyvsp[0] );
        }
#line 1646 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 57:
#line 212 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = GetArcArgs ( yyvsp[-8], yyvsp[-6], yyvsp[-4], yyvsp[-2], yyvsp[0] );
        }
#line 1654 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 58:
#line 217 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1660 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 59:
#line 219 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1669 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 60:
#line 224 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-2], yyvsp[-1] );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1678 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 61:
#line 231 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            auto& left = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-2] );
            auto& center = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-1] );
            auto& right = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] );
            left.reserve ( left.size() + center.size() + right.size() );
            left.insert ( left.end(), center.begin(), center.end() );
            left.insert ( left.end(), right.begin(), right.end() );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1692 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 62:
#line 242 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            auto& left = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-4] );
            auto& center = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-2] );
            auto& right = std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[0] );
            left.reserve ( left.size() + center.size() + right.size() );
            left.insert ( left.end(), center.begin(), center.end() );
            left.insert ( left.end(), right.begin(), right.end() );
            yyval = std::move ( yyvsp[-4] );
        }
#line 1706 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 63:
#line 253 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1712 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 64:
#line 256 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1721 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 65:
#line 261 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-2], yyvsp[-1] );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1730 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 66:
#line 268 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1739 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 67:
#line 274 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-2], yyvsp[-1] );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1748 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 68:
#line 280 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::move ( yyvsp[0] );
        }
#line 1754 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 69:
#line 283 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-1], yyvsp[0] );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1763 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 70:
#line 289 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            Merge ( yyvsp[-2], yyvsp[-1] );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1772 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 71:
#line 295 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::vector<AeonGUI::DrawType> {std::get<AeonGUI::DrawType> ( yyvsp[-1] ), std::get<AeonGUI::DrawType> ( yyvsp[0] ) };
        }
#line 1778 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 72:
#line 297 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::vector<AeonGUI::DrawType> {std::get<AeonGUI::DrawType> ( yyvsp[-2] ), std::get<AeonGUI::DrawType> ( yyvsp[0] ) };
        }
#line 1784 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 73:
#line 301 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::vector<AeonGUI::DrawType> {std::get<AeonGUI::DrawType> ( yyvsp[0] ) };
        }
#line 1792 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 74:
#line 306 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-1] ).emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
            yyval = std::move ( yyvsp[-1] );
        }
#line 1801 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 75:
#line 312 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            std::get<std::vector<AeonGUI::DrawType>> ( yyvsp[-2] ).emplace_back ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) );
            yyval = std::move ( yyvsp[-2] );
        }
#line 1810 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;

    case 77:
#line 321 "C:/Code/AeonGUI/core/parsers/path_data.ypp"
        {
            yyval = std::get<bool> ( std::get<AeonGUI::DrawType> ( yyvsp[0] ) ) ? 1.0 : 0.0;
        }
#line 1818 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"
        break;


#line 1822 "C:/Code/AeonGUI/mingw64/core/path_data_parser.cpp"

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
    YY_SYMBOL_PRINT ( "-> $$ =", yyr1[yyn], &yyval, &yyloc );

    YYPOPSTACK ( yylen );
    yylen = 0;
    YY_STACK_PRINT ( yyss, yyssp );

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
    yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE ( yychar );

    /* If not already recovering from an error, report this error.  */
    if ( !yyerrstatus )
    {
        ++yynerrs;
#if ! YYERROR_VERBOSE
        yyerror ( aPath, YY_ ( "syntax error" ) );
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
        {
            char const *yymsgp = YY_ ( "syntax error" );
            int yysyntax_error_status;
            yysyntax_error_status = YYSYNTAX_ERROR;
            if ( yysyntax_error_status == 0 )
            {
                yymsgp = yymsg;
            }
            else if ( yysyntax_error_status == 1 )
            {
                if ( yymsg != yymsgbuf )
                {
                    YYSTACK_FREE ( yymsg );
                }
                yymsg = ( char * ) YYSTACK_ALLOC ( yymsg_alloc );
                if ( !yymsg )
                {
                    yymsg = yymsgbuf;
                    yymsg_alloc = sizeof yymsgbuf;
                    yysyntax_error_status = 2;
                }
                else
                {
                    yysyntax_error_status = YYSYNTAX_ERROR;
                    yymsgp = yymsg;
                }
            }
            yyerror ( aPath, yymsgp );
            if ( yysyntax_error_status == 2 )
            {
                goto yyexhaustedlab;
            }
        }
# undef YYSYNTAX_ERROR
#endif
    }



    if ( yyerrstatus == 3 )
    {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        if ( yychar <= YYEOF )
        {
            /* Return failure if at end of input.  */
            if ( yychar == YYEOF )
            {
                YYABORT;
            }
        }
        else
        {
            yydestruct ( "Error: discarding",
                         yytoken, &yylval, aPath );
            yychar = YYEMPTY;
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

    for ( ;; )
    {
        yyn = yypact[yystate];
        if ( !yypact_value_is_default ( yyn ) )
        {
            yyn += YYTERROR;
            if ( 0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR )
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
                     yystos[yystate], yyvsp, aPath );
        YYPOPSTACK ( 1 );
        yystate = *yyssp;
        YY_STACK_PRINT ( yyss, yyssp );
    }

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    *++yyvsp = yylval;
    YY_IGNORE_MAYBE_UNINITIALIZED_END


    /* Shift the error token.  */
    YY_SYMBOL_PRINT ( "Shifting", yystos[yyn], yyvsp, yylsp );

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


#if !defined yyoverflow || YYERROR_VERBOSE
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
    if ( yychar != YYEMPTY )
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
                     yystos[*yyssp], yyvsp, aPath );
        YYPOPSTACK ( 1 );
    }
#ifndef yyoverflow
    if ( yyss != yyssa )
    {
        YYSTACK_FREE ( yyss );
    }
#endif
#if YYERROR_VERBOSE
    if ( yymsg != yymsgbuf )
    {
        YYSTACK_FREE ( yymsg );
    }
#endif
    return yyresult;
}
#line 324 "C:/Code/AeonGUI/core/parsers/path_data.ypp"

extern "C"
{
    int derror ( std::vector<AeonGUI::DrawType>& aPath, const char *s )
    {
        std::cerr << s << std::endl;
        return 0;
    }
}
