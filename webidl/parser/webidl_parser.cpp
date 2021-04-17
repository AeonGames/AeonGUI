/* A Bison parser, made by GNU Bison 3.7.6.  */

/* Bison implementation for Yacc-like parsers in C

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

/* Identify Bison output, and Bison version.  */
#define YYBISON 30706

/* Bison version string.  */
#define YYBISON_VERSION "3.7.6"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Substitute the type names.  */
#define YYSTYPE         WEBIDLSTYPE
#define YYLTYPE         WEBIDLLTYPE
/* Substitute the variable and function names.  */
#define yyparse         webidlparse
#define yylex           webidllex
#define yyerror         webidlerror
#define yydebug         webidldebug
#define yynerrs         webidlnerrs
#define yylval          webidllval
#define yychar          webidlchar
#define yylloc          webidllloc

/* First part of user prologue.  */
#line 15 "C:/Code/AeonGUI/webidl/parser/webidl.ypp"

#define YY_NO_UNISTD_H 1
#include <variant>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>

extern int webidllex();
void webidlerror ( const char *s );


#line 95 "C:/Code/AeonGUI/mingw64/webidl/parser/webidl_parser.cpp"

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

#include "webidl_parser.hpp"
/* Symbol kind.  */
enum yysymbol_kind_t
{
    YYSYMBOL_YYEMPTY = -2,
    YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
    YYSYMBOL_YYerror = 1,                    /* error  */
    YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
    YYSYMBOL_integer = 3,                    /* integer  */
    YYSYMBOL_decimal = 4,                    /* decimal  */
    YYSYMBOL_identifier = 5,                 /* identifier  */
    YYSYMBOL_string = 6,                     /* string  */
    YYSYMBOL_null = 7,                       /* null  */
    YYSYMBOL_ellipsis = 8,                   /* ellipsis  */
    YYSYMBOL_async = 9,                      /* async  */
    YYSYMBOL_attribute = 10,                 /* attribute  */
    YYSYMBOL_callback = 11,                  /* callback  */
    YYSYMBOL_CONST = 12,                     /* CONST  */
    YYSYMBOL_constructor = 13,               /* constructor  */
    YYSYMBOL_deleter = 14,                   /* deleter  */
    YYSYMBOL_dictionary = 15,                /* dictionary  */
    YYSYMBOL_ENUM = 16,                      /* ENUM  */
    YYSYMBOL_getter = 17,                    /* getter  */
    YYSYMBOL_includes = 18,                  /* includes  */
    YYSYMBOL_inherit = 19,                   /* inherit  */
    YYSYMBOL_interface = 20,                 /* interface  */
    YYSYMBOL_iterable = 21,                  /* iterable  */
    YYSYMBOL_maplike = 22,                   /* maplike  */
    YYSYMBOL_mixin = 23,                     /* mixin  */
    YYSYMBOL_NAMESPACE = 24,                 /* NAMESPACE  */
    YYSYMBOL_partial = 25,                   /* partial  */
    YYSYMBOL_readonly = 26,                  /* readonly  */
    YYSYMBOL_required = 27,                  /* required  */
    YYSYMBOL_setlike = 28,                   /* setlike  */
    YYSYMBOL_setter = 29,                    /* setter  */
    YYSYMBOL_STATIC = 30,                    /* STATIC  */
    YYSYMBOL_stringifier = 31,               /* stringifier  */
    YYSYMBOL_TYPEDEF = 32,                   /* TYPEDEF  */
    YYSYMBOL_unrestricted = 33,              /* unrestricted  */
    YYSYMBOL_OR = 34,                        /* OR  */
    YYSYMBOL_FLOAT = 35,                     /* FLOAT  */
    YYSYMBOL_DOUBLE = 36,                    /* DOUBLE  */
    YYSYMBOL_TRUEK = 37,                     /* TRUEK  */
    YYSYMBOL_FALSEK = 38,                    /* FALSEK  */
    YYSYMBOL_UNSIGNED = 39,                  /* UNSIGNED  */
    YYSYMBOL_INF = 40,                       /* INF  */
    YYSYMBOL_NEGINF = 41,                    /* NEGINF  */
    YYSYMBOL_NaN = 42,                       /* NaN  */
    YYSYMBOL_optional = 43,                  /* optional  */
    YYSYMBOL_any = 44,                       /* any  */
    YYSYMBOL_other = 45,                     /* other  */
    YYSYMBOL_sequence = 46,                  /* sequence  */
    YYSYMBOL_object = 47,                    /* object  */
    YYSYMBOL_symbol = 48,                    /* symbol  */
    YYSYMBOL_FrozenArray = 49,               /* FrozenArray  */
    YYSYMBOL_ObservableArray = 50,           /* ObservableArray  */
    YYSYMBOL_boolean = 51,                   /* boolean  */
    YYSYMBOL_byte = 52,                      /* byte  */
    YYSYMBOL_octet = 53,                     /* octet  */
    YYSYMBOL_bigint = 54,                    /* bigint  */
    YYSYMBOL_SHORT = 55,                     /* SHORT  */
    YYSYMBOL_LONG = 56,                      /* LONG  */
    YYSYMBOL_Promise = 57,                   /* Promise  */
    YYSYMBOL_record = 58,                    /* record  */
    YYSYMBOL_ArrayBuffer = 59,               /* ArrayBuffer  */
    YYSYMBOL_DataView = 60,                  /* DataView  */
    YYSYMBOL_Int8Array = 61,                 /* Int8Array  */
    YYSYMBOL_Int16Array = 62,                /* Int16Array  */
    YYSYMBOL_Int32Array = 63,                /* Int32Array  */
    YYSYMBOL_Uint8Array = 64,                /* Uint8Array  */
    YYSYMBOL_Uint16Array = 65,               /* Uint16Array  */
    YYSYMBOL_Uint32Array = 66,               /* Uint32Array  */
    YYSYMBOL_Uint8ClampedArray = 67,         /* Uint8ClampedArray  */
    YYSYMBOL_Float32Array = 68,              /* Float32Array  */
    YYSYMBOL_Float64Array = 69,              /* Float64Array  */
    YYSYMBOL_undefined = 70,                 /* undefined  */
    YYSYMBOL_ByteString = 71,                /* ByteString  */
    YYSYMBOL_DOMString = 72,                 /* DOMString  */
    YYSYMBOL_USVString = 73,                 /* USVString  */
    YYSYMBOL_74_ = 74,                       /* '{'  */
    YYSYMBOL_75_ = 75,                       /* '}'  */
    YYSYMBOL_76_ = 76,                       /* ';'  */
    YYSYMBOL_77_ = 77,                       /* ':'  */
    YYSYMBOL_78_ = 78,                       /* '='  */
    YYSYMBOL_79_ = 79,                       /* '['  */
    YYSYMBOL_80_ = 80,                       /* ']'  */
    YYSYMBOL_81_ = 81,                       /* '('  */
    YYSYMBOL_82_ = 82,                       /* ')'  */
    YYSYMBOL_83_ = 83,                       /* ','  */
    YYSYMBOL_84_ = 84,                       /* '<'  */
    YYSYMBOL_85_ = 85,                       /* '>'  */
    YYSYMBOL_86_ = 86,                       /* '?'  */
    YYSYMBOL_YYACCEPT = 87,                  /* $accept  */
    YYSYMBOL_Definitions = 88,               /* Definitions  */
    YYSYMBOL_Definition = 89,                /* Definition  */
    YYSYMBOL_ArgumentNameKeyword = 90,       /* ArgumentNameKeyword  */
    YYSYMBOL_CallbackOrInterfaceOrMixin = 91, /* CallbackOrInterfaceOrMixin  */
    YYSYMBOL_InterfaceOrMixin = 92,          /* InterfaceOrMixin  */
    YYSYMBOL_InterfaceRest = 93,             /* InterfaceRest  */
    YYSYMBOL_Partial = 94,                   /* Partial  */
    YYSYMBOL_PartialDefinition = 95,         /* PartialDefinition  */
    YYSYMBOL_PartialInterfaceOrPartialMixin = 96, /* PartialInterfaceOrPartialMixin  */
    YYSYMBOL_PartialInterfaceRest = 97,      /* PartialInterfaceRest  */
    YYSYMBOL_InterfaceMembers = 98,          /* InterfaceMembers  */
    YYSYMBOL_InterfaceMember = 99,           /* InterfaceMember  */
    YYSYMBOL_PartialInterfaceMembers = 100,  /* PartialInterfaceMembers  */
    YYSYMBOL_PartialInterfaceMember = 101,   /* PartialInterfaceMember  */
    YYSYMBOL_Inheritance = 102,              /* Inheritance  */
    YYSYMBOL_MixinRest = 103,                /* MixinRest  */
    YYSYMBOL_MixinMembers = 104,             /* MixinMembers  */
    YYSYMBOL_MixinMember = 105,              /* MixinMember  */
    YYSYMBOL_IncludesStatement = 106,        /* IncludesStatement  */
    YYSYMBOL_CallbackRestOrInterface = 107,  /* CallbackRestOrInterface  */
    YYSYMBOL_CallbackInterfaceMembers = 108, /* CallbackInterfaceMembers  */
    YYSYMBOL_CallbackInterfaceMember = 109,  /* CallbackInterfaceMember  */
    YYSYMBOL_Const = 110,                    /* Const  */
    YYSYMBOL_ConstValue = 111,               /* ConstValue  */
    YYSYMBOL_BooleanLiteral = 112,           /* BooleanLiteral  */
    YYSYMBOL_FloatLiteral = 113,             /* FloatLiteral  */
    YYSYMBOL_ConstType = 114,                /* ConstType  */
    YYSYMBOL_ReadOnlyMember = 115,           /* ReadOnlyMember  */
    YYSYMBOL_ReadOnlyMemberRest = 116,       /* ReadOnlyMemberRest  */
    YYSYMBOL_ReadWriteAttribute = 117,       /* ReadWriteAttribute  */
    YYSYMBOL_InheritAttribute = 118,         /* InheritAttribute  */
    YYSYMBOL_AttributeRest = 119,            /* AttributeRest  */
    YYSYMBOL_AttributeName = 120,            /* AttributeName  */
    YYSYMBOL_AttributeNameKeyword = 121,     /* AttributeNameKeyword  */
    YYSYMBOL_OptionalReadOnly = 122,         /* OptionalReadOnly  */
    YYSYMBOL_DefaultValue = 123,             /* DefaultValue  */
    YYSYMBOL_Operation = 124,                /* Operation  */
    YYSYMBOL_RegularOperation = 125,         /* RegularOperation  */
    YYSYMBOL_SpecialOperation = 126,         /* SpecialOperation  */
    YYSYMBOL_Special = 127,                  /* Special  */
    YYSYMBOL_OperationRest = 128,            /* OperationRest  */
    YYSYMBOL_OptionalOperationName = 129,    /* OptionalOperationName  */
    YYSYMBOL_OperationName = 130,            /* OperationName  */
    YYSYMBOL_OperationNameKeyword = 131,     /* OperationNameKeyword  */
    YYSYMBOL_ArgumentList = 132,             /* ArgumentList  */
    YYSYMBOL_Arguments = 133,                /* Arguments  */
    YYSYMBOL_Argument = 134,                 /* Argument  */
    YYSYMBOL_ArgumentName = 135,             /* ArgumentName  */
    YYSYMBOL_Ellipsis = 136,                 /* Ellipsis  */
    YYSYMBOL_Constructor = 137,              /* Constructor  */
    YYSYMBOL_Stringifier = 138,              /* Stringifier  */
    YYSYMBOL_StringifierRest = 139,          /* StringifierRest  */
    YYSYMBOL_StaticMember = 140,             /* StaticMember  */
    YYSYMBOL_StaticMemberRest = 141,         /* StaticMemberRest  */
    YYSYMBOL_Iterable = 142,                 /* Iterable  */
    YYSYMBOL_OptionalType = 143,             /* OptionalType  */
    YYSYMBOL_AsyncIterable = 144,            /* AsyncIterable  */
    YYSYMBOL_OptionalArgumentList = 145,     /* OptionalArgumentList  */
    YYSYMBOL_ReadWriteMaplike = 146,         /* ReadWriteMaplike  */
    YYSYMBOL_MaplikeRest = 147,              /* MaplikeRest  */
    YYSYMBOL_ReadWriteSetlike = 148,         /* ReadWriteSetlike  */
    YYSYMBOL_SetlikeRest = 149,              /* SetlikeRest  */
    YYSYMBOL_Namespace = 150,                /* Namespace  */
    YYSYMBOL_NamespaceMembers = 151,         /* NamespaceMembers  */
    YYSYMBOL_NamespaceMember = 152,          /* NamespaceMember  */
    YYSYMBOL_Dictionary = 153,               /* Dictionary  */
    YYSYMBOL_DictionaryMembers = 154,        /* DictionaryMembers  */
    YYSYMBOL_DictionaryMember = 155,         /* DictionaryMember  */
    YYSYMBOL_DictionaryMemberRest = 156,     /* DictionaryMemberRest  */
    YYSYMBOL_PartialDictionary = 157,        /* PartialDictionary  */
    YYSYMBOL_Default = 158,                  /* Default  */
    YYSYMBOL_Enum = 159,                     /* Enum  */
    YYSYMBOL_EnumValueList = 160,            /* EnumValueList  */
    YYSYMBOL_EnumValueListComma = 161,       /* EnumValueListComma  */
    YYSYMBOL_EnumValueListString = 162,      /* EnumValueListString  */
    YYSYMBOL_CallbackRest = 163,             /* CallbackRest  */
    YYSYMBOL_Typedef = 164,                  /* Typedef  */
    YYSYMBOL_Type = 165,                     /* Type  */
    YYSYMBOL_TypeWithExtendedAttributes = 166, /* TypeWithExtendedAttributes  */
    YYSYMBOL_SingleType = 167,               /* SingleType  */
    YYSYMBOL_UnionType = 168,                /* UnionType  */
    YYSYMBOL_UnionMemberType = 169,          /* UnionMemberType  */
    YYSYMBOL_UnionMemberTypes = 170,         /* UnionMemberTypes  */
    YYSYMBOL_DistinguishableType = 171,      /* DistinguishableType  */
    YYSYMBOL_PrimitiveType = 172,            /* PrimitiveType  */
    YYSYMBOL_UnrestrictedFloatType = 173,    /* UnrestrictedFloatType  */
    YYSYMBOL_FloatType = 174,                /* FloatType  */
    YYSYMBOL_UnsignedIntegerType = 175,      /* UnsignedIntegerType  */
    YYSYMBOL_IntegerType = 176,              /* IntegerType  */
    YYSYMBOL_OptionalLong = 177,             /* OptionalLong  */
    YYSYMBOL_StringType = 178,               /* StringType  */
    YYSYMBOL_PromiseType = 179,              /* PromiseType  */
    YYSYMBOL_RecordType = 180,               /* RecordType  */
    YYSYMBOL_Null = 181,                     /* Null  */
    YYSYMBOL_BufferRelatedType = 182,        /* BufferRelatedType  */
    YYSYMBOL_ExtendedAttributeList = 183,    /* ExtendedAttributeList  */
    YYSYMBOL_ExtendedAttributes = 184,       /* ExtendedAttributes  */
    YYSYMBOL_StringLiteral = 185,            /* StringLiteral  */
    YYSYMBOL_StringLiteralList = 186,        /* StringLiteralList  */
    YYSYMBOL_ExtendedAttribute = 187,        /* ExtendedAttribute  */
    YYSYMBOL_IdentifierList = 188,           /* IdentifierList  */
    YYSYMBOL_Identifiers = 189,              /* Identifiers  */
    YYSYMBOL_ExtendedAttributeNoArgs = 190,  /* ExtendedAttributeNoArgs  */
    YYSYMBOL_ExtendedAttributeArgList = 191, /* ExtendedAttributeArgList  */
    YYSYMBOL_ExtendedAttributeIdent = 192,   /* ExtendedAttributeIdent  */
    YYSYMBOL_ExtendedAttributeIdentList = 193, /* ExtendedAttributeIdentList  */
    YYSYMBOL_ExtendedAttributeNamedArgList = 194, /* ExtendedAttributeNamedArgList  */
    YYSYMBOL_ExtendedAttributeStringLiteral = 195, /* ExtendedAttributeStringLiteral  */
    YYSYMBOL_ExtendedAttributeStringLiteralList = 196 /* ExtendedAttributeStringLiteralList  */
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

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
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
typedef yytype_int16 yy_state_t;

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
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
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

#if 1

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
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined WEBIDLLTYPE_IS_TRIVIAL && WEBIDLLTYPE_IS_TRIVIAL \
             && defined WEBIDLSTYPE_IS_TRIVIAL && WEBIDLSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
    yy_state_t yyss_alloc;
    YYSTYPE yyvs_alloc;
    YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

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
#define YYFINAL  13
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   971

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  87
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  249
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  432

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   328


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
    81,    82,     2,     2,    83,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,    77,    76,
    84,    78,    85,    86,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,    79,     2,    80,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,    74,     2,    75,     2,     2,     2,     2,
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
    2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
    5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
    15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
    25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
    35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
    45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
    55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
    65,    66,    67,    68,    69,    70,    71,    72,    73
};

#if WEBIDLDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
    0,   103,   103,   103,   107,   108,   109,   110,   111,   112,
    113,   117,   118,   119,   120,   121,   122,   123,   124,   125,
    126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
    136,   137,   138,   139,   140,   141,   145,   146,   150,   151,
    154,   158,   161,   163,   167,   168,   172,   176,   176,   180,
    181,   185,   185,   189,   190,   191,   192,   193,   194,   195,
    196,   197,   198,   199,   203,   203,   207,   211,   211,   215,
    216,   217,   218,   222,   226,   227,   231,   231,   235,   236,
    240,   244,   245,   246,   250,   251,   255,   256,   257,   258,
    262,   263,   267,   271,   272,   273,   277,   281,   285,   289,
    290,   294,   295,   299,   299,   303,   304,   305,   306,   307,
    311,   312,   316,   320,   324,   325,   326,   330,   334,   334,
    338,   339,   343,   347,   347,   351,   351,   355,   356,   360,
    361,   365,   365,   369,   373,   377,   379,   383,   387,   392,
    396,   396,   400,   404,   404,   408,   411,   415,   418,   421,
    424,   424,   428,   429,   433,   437,   437,   441,   445,   446,
    450,   454,   454,   458,   462,   466,   466,   470,   470,   474,
    478,   482,   483,   487,   491,   492,   493,   497,   501,   502,
    506,   506,   510,   511,   512,   513,   514,   515,   516,   517,
    518,   519,   523,   524,   525,   526,   527,   528,   529,   533,
    538,   539,   543,   544,   548,   549,   553,   553,   557,   558,
    559,   563,   567,   571,   571,   575,   576,   577,   578,   579,
    580,   581,   582,   583,   584,   585,   589,   589,   593,   593,
    597,   601,   602,   607,   608,   609,   610,   611,   612,   613,
    617,   621,   621,   625,   629,   633,   637,   641,   645,   649
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name ( yysymbol_kind_t yysymbol ) YY_ATTRIBUTE_UNUSED;

static const char *
yysymbol_name ( yysymbol_kind_t yysymbol )
{
    static const char *const yy_sname[] =
    {
        "end of file", "error", "invalid token", "integer", "decimal",
        "identifier", "string", "null", "ellipsis", "async", "attribute",
        "callback", "CONST", "constructor", "deleter", "dictionary", "ENUM",
        "getter", "includes", "inherit", "interface", "iterable", "maplike",
        "mixin", "NAMESPACE", "partial", "readonly", "required", "setlike",
        "setter", "STATIC", "stringifier", "TYPEDEF", "unrestricted", "OR",
        "FLOAT", "DOUBLE", "TRUEK", "FALSEK", "UNSIGNED", "INF", "NEGINF", "NaN",
        "optional", "any", "other", "sequence", "object", "symbol",
        "FrozenArray", "ObservableArray", "boolean", "byte", "octet", "bigint",
        "SHORT", "LONG", "Promise", "record", "ArrayBuffer", "DataView",
        "Int8Array", "Int16Array", "Int32Array", "Uint8Array", "Uint16Array",
        "Uint32Array", "Uint8ClampedArray", "Float32Array", "Float64Array",
        "undefined", "ByteString", "DOMString", "USVString", "'{'", "'}'", "';'",
        "':'", "'='", "'['", "']'", "'('", "')'", "','", "'<'", "'>'", "'?'",
        "$accept", "Definitions", "Definition", "ArgumentNameKeyword",
        "CallbackOrInterfaceOrMixin", "InterfaceOrMixin", "InterfaceRest",
        "Partial", "PartialDefinition", "PartialInterfaceOrPartialMixin",
        "PartialInterfaceRest", "InterfaceMembers", "InterfaceMember",
        "PartialInterfaceMembers", "PartialInterfaceMember", "Inheritance",
        "MixinRest", "MixinMembers", "MixinMember", "IncludesStatement",
        "CallbackRestOrInterface", "CallbackInterfaceMembers",
        "CallbackInterfaceMember", "Const", "ConstValue", "BooleanLiteral",
        "FloatLiteral", "ConstType", "ReadOnlyMember", "ReadOnlyMemberRest",
        "ReadWriteAttribute", "InheritAttribute", "AttributeRest",
        "AttributeName", "AttributeNameKeyword", "OptionalReadOnly",
        "DefaultValue", "Operation", "RegularOperation", "SpecialOperation",
        "Special", "OperationRest", "OptionalOperationName", "OperationName",
        "OperationNameKeyword", "ArgumentList", "Arguments", "Argument",
        "ArgumentName", "Ellipsis", "Constructor", "Stringifier",
        "StringifierRest", "StaticMember", "StaticMemberRest", "Iterable",
        "OptionalType", "AsyncIterable", "OptionalArgumentList",
        "ReadWriteMaplike", "MaplikeRest", "ReadWriteSetlike", "SetlikeRest",
        "Namespace", "NamespaceMembers", "NamespaceMember", "Dictionary",
        "DictionaryMembers", "DictionaryMember", "DictionaryMemberRest",
        "PartialDictionary", "Default", "Enum", "EnumValueList",
        "EnumValueListComma", "EnumValueListString", "CallbackRest", "Typedef",
        "Type", "TypeWithExtendedAttributes", "SingleType", "UnionType",
        "UnionMemberType", "UnionMemberTypes", "DistinguishableType",
        "PrimitiveType", "UnrestrictedFloatType", "FloatType",
        "UnsignedIntegerType", "IntegerType", "OptionalLong", "StringType",
        "PromiseType", "RecordType", "Null", "BufferRelatedType",
        "ExtendedAttributeList", "ExtendedAttributes", "StringLiteral",
        "StringLiteralList", "ExtendedAttribute", "IdentifierList",
        "Identifiers", "ExtendedAttributeNoArgs", "ExtendedAttributeArgList",
        "ExtendedAttributeIdent", "ExtendedAttributeIdentList",
        "ExtendedAttributeNamedArgList", "ExtendedAttributeStringLiteral",
        "ExtendedAttributeStringLiteralList", YY_NULLPTR
    };
    return yy_sname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
    0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
    265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
    275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
    285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
    295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
    305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
    315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
    325,   326,   327,   328,   123,   125,    59,    58,    61,    91,
    93,    40,    41,    44,    60,    62,    63
};
#endif

#define YYPACT_NINF (-205)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-157)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    5,    29,    42,    94,    20,   -34,  -205,  -205,  -205,  -205,
    -205,  -205,  -205,  -205,    67,    18,    92,   112,    63,   115,
    6,    46,     5,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    8,    24,    29,    48,   130,    69,   141,  -205,  -205,    71,
    75,    71,   145,  -205,  -205,  -205,    78,    64,  -205,  -205,
    146,   800,  -205,    73,  -205,    68,  -205,    76,    74,   523,
    -34,  -205,    83,   800,    86,   156,    88,   157,    90,    91,
    -50,    93,   151,  -205,  -205,    95,    82,    77,    89,  -205,
    85,    82,    82,    97,    98,  -205,  -205,  -205,  -205,  -205,
    114,    99,   101,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,    50,  -205,
    -205,    82,  -205,    82,  -205,  -205,  -205,    82,  -205,    82,
    82,    24,    96,   103,   105,   106,  -205,    46,  -205,    46,
    164,  -205,  -205,   108,   -42,  -205,     1,   107,   100,     4,
    25,   102,   592,    36,   168,  -205,  -205,  -205,  -205,  -205,
    -205,    77,  -205,    46,  -205,  -205,    46,    46,  -205,  -205,
    800,    51,    82,   140,   869,  -205,  -205,  -205,  -205,  -205,
    109,   171,  -205,   172,  -205,  -205,    74,   938,  -205,   938,
    24,   117,   662,   119,     1,   731,   189,  -205,   121,   123,
    314,   124,   454,   125,   190,  -205,   -50,    22,   127,   384,
    129,  -205,   120,   122,   126,   131,   132,  -205,    50,  -205,
    -205,    96,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,   128,  -205,   135,   133,    11,   -42,  -205,  -205,   134,
    -205,    46,  -205,   199,   107,  -205,  -205,   142,   200,    46,
    139,  -205,  -205,   190,   138,   143,    49,   144,  -205,   197,
    -15,     4,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,   800,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    -205,   148,  -205,    25,  -205,   190,  -205,  -205,  -205,  -205,
    -205,  -205,  -205,  -205,   149,  -205,  -205,   150,    36,     1,
    82,    82,    82,  -205,    46,   191,  -205,    14,  -205,   153,
    -205,  -205,   227,  -205,  -205,  -205,   228,   128,  -205,  -205,
    152,    19,    24,  -205,    46,    46,  -205,  -205,  -205,  -205,
    46,   190,  -205,  -205,   190,  -205,  -205,  -205,  -205,  -205,
    -205,    24,  -205,  -205,   159,  -205,  -205,  -205,   158,    50,
    155,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
    160,   161,  -205,  -205,  -205,  -205,  -205,   166,   169,   173,
    46,  -205,  -205,  -205,   174,  -205,   165,   175,   177,   163,
    800,   800,   179,   178,  -205,   191,  -205,  -205,  -205,    54,
    -205,  -205,   175,  -205,   180,    46,   170,    46,   181,  -205,
    -205,   186,  -205,  -205,   187,   182,  -205,  -205,   188,   183,
    -205,  -205,  -205,   184,  -205,   193,    24,   194,  -205,   192,
    -205,  -205
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
    227,     0,     0,     0,   243,   229,   233,   234,   235,   236,
    237,   238,   239,     1,     0,     0,     0,     0,     0,     0,
    0,   227,   227,     4,     6,    10,     5,     7,     8,     9,
    0,   227,     0,     0,     0,     0,     0,    36,    74,    65,
    0,    65,     0,    37,    38,    39,     0,     0,    41,    43,
    0,     0,     2,   245,   230,     0,   248,     0,   126,     0,
    229,   226,     0,     0,     0,     0,     0,     0,     0,     0,
    227,     0,     0,    44,    45,     0,   214,     0,     0,   175,
    0,   214,   214,     0,     0,   195,   196,   197,   198,   204,
    207,     0,     0,   215,   216,   217,   218,   219,   220,   221,
    222,   223,   224,   225,   194,   208,   209,   210,   227,   173,
    171,   214,   174,   214,   193,   192,   203,   214,   176,   214,
    214,   227,   242,   232,     0,     0,   244,   227,   123,   227,
    132,   228,    73,     0,   227,    64,   227,   166,     0,   227,
    227,     0,     0,   227,     0,    42,   170,   213,   184,   200,
    201,     0,   202,   227,   186,   187,   227,   227,   206,   205,
    0,     0,   214,     0,     0,   172,   182,   183,   191,   188,
    0,     0,   240,     0,   249,   246,   126,     0,   131,     0,
    227,     0,     0,     0,   227,     0,   168,   164,     0,     0,
    0,     0,   104,     0,     0,   152,   227,   119,     0,     0,
    0,   199,     0,     0,     0,     0,     0,   179,   227,   178,
    247,   242,   231,   125,   130,    11,    12,    13,    14,    15,
    16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
    26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
    129,   162,   128,     0,     0,     0,   227,    78,    79,     0,
    155,   227,   157,     0,   166,   165,   163,     0,     0,   227,
    0,   116,   114,     0,     0,     0,     0,     0,   115,   104,
    104,   227,    49,    53,    59,    60,    63,    96,    54,   110,
    111,     0,    50,    55,    56,    57,    58,    61,   145,    62,
    147,     0,   103,   227,    69,     0,    70,    71,   149,   153,
    150,   121,   122,   112,     0,   118,   120,     0,   227,   227,
    214,   214,   214,   211,   227,   181,   241,     0,   127,     0,
    75,    91,     0,    90,    76,   154,     0,   162,   167,    40,
    0,     0,   227,    97,   227,   227,    92,    93,    94,    95,
    227,     0,   137,   136,     0,   134,    47,   113,    66,    67,
    72,   227,    46,    51,     0,   185,   189,   190,     0,   227,
    0,    83,    86,   106,   109,    84,    85,    88,    87,    89,
    0,     0,   105,    81,    82,   161,   169,     0,     0,     0,
    227,   100,   101,   102,     0,    99,     0,   141,     0,     0,
    0,     0,     0,     0,   212,   181,   177,   108,   107,     0,
    158,   159,   141,    98,     0,   227,     0,   227,     0,   138,
    135,     0,   160,   180,     0,     0,   133,   140,     0,     0,
    148,   117,    80,   144,   139,     0,   227,     0,   146,     0,
    142,   143
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -205,   220,  -205,  -205,  -205,  -205,  -205,  -205,  -205,  -205,
        -205,    -5,  -205,   -36,    79,   232,   229,   -18,  -205,  -205,
        -205,    31,  -205,  -110,  -119,  -205,  -205,  -205,  -205,  -205,
        -205,  -205,  -188,  -205,  -205,  -137,  -205,  -205,  -139,  -205,
        -205,  -205,  -205,  -205,  -205,  -120,   111,   154,   104,  -205,
        -205,    87,  -205,  -205,  -205,  -205,  -117,  -205,  -205,  -205,
        16,  -205,    23,   268,   116,  -205,  -205,  -172,  -205,  -205,
        -205,   -37,  -205,  -205,    37,  -205,  -205,  -205,   -44,  -121,
        -205,  -106,  -204,  -103,   136,    52,  -205,   147,  -205,   216,
        -205,   176,  -205,  -205,   -72,  -205,     0,   235,   266,   137,
        267,  -205,   110,  -205,  -205,  -205,  -205,  -205,  -205,  -205
    };

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
    0,     2,    22,   240,    23,    43,    44,    24,    48,    72,
    73,   189,   271,   198,   272,    66,    45,   191,   293,    25,
    37,   181,   246,   273,   372,   373,   374,   322,   274,   336,
    275,   276,   277,   384,   385,   295,   375,   278,   279,   280,
    281,   303,   304,   305,   306,    57,   128,    58,   241,   179,
    282,   283,   345,   284,   342,   285,   406,   286,   427,   287,
    288,   289,   290,    26,   141,   196,    27,   183,   184,   252,
    145,   318,    28,   138,   187,   255,    38,    29,   197,    50,
    110,   111,   163,   360,   112,   113,   114,   151,   115,   116,
    159,   117,   118,   119,   148,   120,    51,    33,   123,   124,
    5,   125,   172,     6,     7,     8,     9,    10,    11,    12
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
    3,   170,   162,   195,   315,    -3,   299,   109,   177,   154,
    155,   292,   250,    53,    54,   130,   321,   361,   362,   133,
    363,   364,     3,    35,   381,  -151,    47,   301,   382,     1,
    19,    59,   202,   -77,     4,   203,   204,     1,    36,   165,
    302,   166,    13,   248,    77,   167,   383,   168,   169,    32,
    78,   365,   366,   296,   367,   368,   369,   361,   362,   259,
    243,   343,    85,    86,    87,    88,    89,    90,    41,    71,
    142,   265,   247,   122,    54,   333,  -156,   267,   337,   -48,
    1,   104,   294,     1,     1,    34,    42,    42,   370,    55,
    207,   365,   366,   371,   367,   368,   369,    39,    30,    14,
    -68,    31,   162,     1,     1,    15,  -124,   350,   164,    16,
    17,   -52,   149,   150,    18,     1,   205,    40,    19,    20,
    46,    59,   105,   106,   107,     1,    21,    59,    61,     1,
    326,   108,   341,   344,   182,    62,   185,   354,   331,   190,
    192,   253,   347,   199,    89,    90,    64,    63,    65,    67,
    69,    75,    70,   390,   121,   395,   391,   127,   126,   132,
    134,   135,   136,   137,   139,   140,   144,   143,   147,   153,
    158,   146,   178,   200,   208,   188,   211,   193,    54,   171,
    59,   156,   157,   160,   185,   161,   173,   174,   175,   180,
    186,   210,   244,   358,   249,   254,   142,   256,   257,   291,
    259,   298,   307,   309,   327,   310,   317,   311,   164,   320,
    325,   312,   386,   387,   388,   314,   313,   319,   329,   389,
    332,   330,   334,   292,   348,   359,   352,   335,   340,   376,
    351,   392,   377,   378,   393,   397,   380,   396,   355,   356,
    357,   398,    52,   394,   399,   400,   182,   404,   408,   401,
    403,   409,   410,   162,   412,   418,   416,   420,   405,   402,
    407,   411,   421,   422,   424,   426,   346,   423,   425,   428,
    430,   190,   353,    68,   431,   349,    74,   324,   308,   297,
    414,   176,   338,   242,   417,   415,   419,   213,    49,   339,
    379,   328,   413,   192,   152,   131,    56,   323,   201,    60,
    209,     0,     0,     0,     0,     0,   429,     0,   199,   185,
    212,     0,   300,     0,     0,     0,     0,     0,     0,    76,
    0,   316,     0,   258,   259,     0,   245,   260,   261,     0,
    0,   262,    59,   263,     0,   264,   265,   206,     0,     0,
    266,     0,   267,   268,   269,   270,     0,    77,     0,     0,
    0,    59,     0,    78,     0,     0,     0,     0,    79,   164,
    80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
    90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
    100,   101,   102,   103,   104,   105,   106,   107,     0,    76,
    0,     0,     0,   258,   259,   108,   245,     0,   261,     0,
    0,   262,     0,   263,     0,   264,   265,     0,     0,     0,
    266,     0,   267,   268,   269,   270,     0,    77,     0,     0,
    0,     0,     0,    78,     0,     0,    59,     0,    79,     0,
    80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
    90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
    100,   101,   102,   103,   104,   105,   106,   107,     0,    76,
    0,     0,     0,     0,     0,   108,   245,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    292,     0,     0,     0,     0,   270,     0,    77,     0,     0,
    0,     0,     0,    78,     0,     0,     0,     0,    79,     0,
    80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
    90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
    100,   101,   102,   103,   104,   105,   106,   107,    76,     0,
    0,     0,     0,     0,     0,   108,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,    77,     0,     0,     0,
    0,     0,    78,     0,     0,     0,   129,    79,     0,    80,
    81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
    91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
    101,   102,   103,   104,   105,   106,   107,    76,     0,     0,
    0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,   194,     0,
    0,     0,     0,     0,     0,    77,     0,     0,     0,     0,
    0,    78,     0,     0,     0,     0,    79,     0,    80,    81,
    82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
    92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
    102,   103,   104,   105,   106,   107,     0,    76,     0,     0,
    0,     0,     0,   108,   245,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,    77,     0,     0,     0,     0,
    0,    78,     0,     0,     0,     0,    79,     0,    80,    81,
    82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
    92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
    102,   103,   104,   105,   106,   107,    76,     0,     0,     0,
    0,     0,     0,   108,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,   251,     0,
    0,     0,     0,     0,    77,     0,     0,     0,     0,     0,
    78,     0,     0,     0,     0,    79,     0,    80,    81,    82,
    83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
    93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
    103,   104,   105,   106,   107,    76,     0,     0,     0,     0,
    0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,    77,     0,     0,     0,     0,     0,    78,
    0,     0,     0,     0,    79,     0,    80,    81,    82,    83,
    84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
    94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
    104,   105,   106,   107,    76,     0,     0,     0,     0,     0,
    0,   108,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    0,     0,    77,     0,     0,     0,     0,     0,    78,     0,
    0,     0,     0,     0,     0,    80,    81,    82,    83,    84,
    85,    86,    87,    88,    89,    90,     0,    92,    93,    94,
    95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
    105,   106,   107,   214,     0,     0,     0,   215,   216,   217,
    218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
    228,   229,   230,   231,   232,   233,   234,   235,   236,   237,
    238,   239
};

static const yytype_int16 yycheck[] =
{
    0,   121,   108,   142,   208,     0,   194,    51,   129,    81,
    82,    26,   184,     5,     6,    59,     5,     3,     4,    63,
    6,     7,    22,     5,     5,    75,    20,     5,     9,    79,
    24,    31,   153,    75,     5,   156,   157,    79,    20,   111,
    18,   113,     0,   182,    33,   117,    27,   119,   120,    83,
    39,    37,    38,   192,    40,    41,    42,     3,     4,    10,
    180,    76,    51,    52,    53,    54,    55,    56,     5,     5,
    70,    22,   182,     5,     6,   263,    75,    28,   266,    75,
    79,    70,   192,    79,    79,    18,    23,    23,    74,    81,
    162,    37,    38,    79,    40,    41,    42,     5,    78,     5,
    75,    81,   208,    79,    79,    11,    82,   295,   108,    15,
    16,    75,    35,    36,    20,    79,   160,     5,    24,    25,
    5,   121,    71,    72,    73,    79,    32,   127,    80,    79,
    251,    81,   269,   270,   134,     5,   136,   309,   259,   139,
    140,   185,   281,   143,    55,    56,     5,    78,    77,    74,
    5,     5,    74,   341,    81,   359,   344,    83,    82,    76,
    74,     5,    74,     6,    74,    74,    15,    74,    86,    84,
    56,    76,     8,     5,    34,    75,     5,    75,     6,    83,
    180,    84,    84,    84,   184,    84,    83,    82,    82,    81,
    83,    82,    75,   314,    75,     6,   196,    76,    75,    75,
    10,    76,    75,    74,     5,    85,    78,    85,   208,    76,
    76,    85,   332,   334,   335,    83,    85,    82,    76,   340,
    81,    21,    84,    26,    76,    34,    76,    84,    84,    76,
    81,   351,     5,     5,    75,    75,    84,    82,   310,   311,
    312,    80,    22,    85,    78,    76,   246,    82,    85,    76,
    76,   390,   391,   359,    76,    85,    76,    76,    83,   380,
    83,    82,    76,    76,    76,    81,   271,    85,    85,    76,
    76,   271,   308,    41,    82,   293,    47,   246,   199,   192,
    399,   127,   266,   179,   405,   402,   407,   176,    20,   266,
    327,   254,   395,   293,    78,    60,    30,   245,   151,    32,
    164,    -1,    -1,    -1,    -1,    -1,   426,    -1,   308,   309,
    173,    -1,   196,    -1,    -1,    -1,    -1,    -1,    -1,     5,
    -1,   211,    -1,     9,    10,    -1,    12,    13,    14,    -1,
    -1,    17,   332,    19,    -1,    21,    22,   161,    -1,    -1,
    26,    -1,    28,    29,    30,    31,    -1,    33,    -1,    -1,
    -1,   351,    -1,    39,    -1,    -1,    -1,    -1,    44,   359,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    -1,     5,
    -1,    -1,    -1,     9,    10,    81,    12,    -1,    14,    -1,
    -1,    17,    -1,    19,    -1,    21,    22,    -1,    -1,    -1,
    26,    -1,    28,    29,    30,    31,    -1,    33,    -1,    -1,
    -1,    -1,    -1,    39,    -1,    -1,   426,    -1,    44,    -1,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    -1,     5,
    -1,    -1,    -1,    -1,    -1,    81,    12,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    26,    -1,    -1,    -1,    -1,    31,    -1,    33,    -1,    -1,
    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,    -1,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,     5,    -1,
    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    33,    -1,    -1,    -1,
    -1,    -1,    39,    -1,    -1,    -1,    43,    44,    -1,    46,
    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
    57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,     5,    -1,    -1,
    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    -1,
    -1,    -1,    -1,    -1,    -1,    33,    -1,    -1,    -1,    -1,
    -1,    39,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
    58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
    68,    69,    70,    71,    72,    73,    -1,     5,    -1,    -1,
    -1,    -1,    -1,    81,    12,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    33,    -1,    -1,    -1,    -1,
    -1,    39,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
    58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
    68,    69,    70,    71,    72,    73,     5,    -1,    -1,    -1,
    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    27,    -1,
    -1,    -1,    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,
    39,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,
    49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
    59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
    69,    70,    71,    72,    73,     5,    -1,    -1,    -1,    -1,
    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,    39,
    -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
    50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
    60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
    70,    71,    72,    73,     5,    -1,    -1,    -1,    -1,    -1,
    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,    39,    -1,
    -1,    -1,    -1,    -1,    -1,    46,    47,    48,    49,    50,
    51,    52,    53,    54,    55,    56,    -1,    58,    59,    60,
    61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
    71,    72,    73,     5,    -1,    -1,    -1,     9,    10,    11,
    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
    22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
    32,    33
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
    0,    79,    88,   183,     5,   187,   190,   191,   192,   193,
    194,   195,   196,     0,     5,    11,    15,    16,    20,    24,
    25,    32,    89,    91,    94,   106,   150,   153,   159,   164,
    78,    81,    83,   184,    18,     5,    20,   107,   163,     5,
    5,     5,    23,    92,    93,   103,     5,    20,    95,   150,
    166,   183,    88,     5,     6,    81,   185,   132,   134,   183,
    187,    80,     5,    78,     5,    77,   102,    74,   102,     5,
    74,     5,    96,    97,   103,     5,     5,    33,    39,    44,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    81,   165,
    167,   168,   171,   172,   173,   175,   176,   178,   179,   180,
    182,    81,     5,   185,   186,   188,    82,    83,   133,    43,
    165,   184,    76,   165,    74,     5,    74,     6,   160,    74,
    74,   151,   183,    74,    15,   157,    76,    86,   181,    35,
    36,   174,   176,    84,   181,   181,    84,    84,    56,   177,
    84,    84,   168,   169,   183,   181,   181,   181,   181,   181,
    132,    83,   189,    83,    82,    82,   134,   166,     8,   136,
    81,   108,   183,   154,   155,   183,    83,   161,    75,    98,
    183,   104,   183,    75,    26,   125,   152,   165,   100,   183,
    5,   174,   166,   166,   166,   165,   178,   181,    34,   171,
    82,     5,   186,   133,     5,     9,    10,    11,    12,    13,
    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
    24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
    90,   135,   135,   132,    75,    12,   109,   110,   125,    75,
    154,    27,   156,   165,     6,   162,    76,    75,     9,    10,
    13,    14,    17,    19,    21,    22,    26,    28,    29,    30,
    31,    99,   101,   110,   115,   117,   118,   119,   124,   125,
    126,   127,   137,   138,   140,   142,   144,   146,   147,   148,
    149,    75,    26,   105,   110,   122,   125,   138,    76,   119,
    151,     5,    18,   128,   129,   130,   131,    75,   101,    74,
    85,    85,    85,    85,    83,   169,   189,    78,   158,    82,
    76,     5,   114,   172,   108,    76,   166,     5,   161,    76,
    21,   166,    81,   119,    84,    84,   116,   119,   147,   149,
    84,   122,   141,    76,   122,   139,    98,   125,    76,   104,
    119,    81,    76,   100,   154,   181,   181,   181,   166,    34,
    170,     3,     4,     6,     7,    37,    38,    40,    41,    42,
    74,    79,   111,   112,   113,   123,    76,     5,     5,   158,
    84,     5,     9,    27,   120,   121,   132,   166,   166,   166,
    119,   119,   132,    75,    85,   169,    82,    75,    80,    78,
    76,    76,   166,    76,    82,    83,   143,    83,    85,   125,
    125,    82,    76,   170,   111,   143,    76,   166,    85,   166,
    76,    76,    76,    85,    76,    85,    81,   145,    76,   132,
    76,    82
};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
    0,    87,    88,    88,    89,    89,    89,    89,    89,    89,
    89,    90,    90,    90,    90,    90,    90,    90,    90,    90,
    90,    90,    90,    90,    90,    90,    90,    90,    90,    90,
    90,    90,    90,    90,    90,    90,    91,    91,    92,    92,
    93,    94,    95,    95,    96,    96,    97,    98,    98,    99,
    99,   100,   100,   101,   101,   101,   101,   101,   101,   101,
    101,   101,   101,   101,   102,   102,   103,   104,   104,   105,
    105,   105,   105,   106,   107,   107,   108,   108,   109,   109,
    110,   111,   111,   111,   112,   112,   113,   113,   113,   113,
    114,   114,   115,   116,   116,   116,   117,   118,   119,   120,
    120,   121,   121,   122,   122,   123,   123,   123,   123,   123,
    124,   124,   125,   126,   127,   127,   127,   128,   129,   129,
    130,   130,   131,   132,   132,   133,   133,   134,   134,   135,
    135,   136,   136,   137,   138,   139,   139,   140,   141,   142,
    143,   143,   144,   145,   145,   146,   147,   148,   149,   150,
    151,   151,   152,   152,   153,   154,   154,   155,   156,   156,
    157,   158,   158,   159,   160,   161,   161,   162,   162,   163,
    164,   165,   165,   166,   167,   167,   167,   168,   169,   169,
    170,   170,   171,   171,   171,   171,   171,   171,   171,   171,
    171,   171,   172,   172,   172,   172,   172,   172,   172,   173,
    174,   174,   175,   175,   176,   176,   177,   177,   178,   178,
    178,   179,   180,   181,   181,   182,   182,   182,   182,   182,
    182,   182,   182,   182,   182,   182,   183,   183,   184,   184,
    185,   186,   186,   187,   187,   187,   187,   187,   187,   187,
    188,   189,   189,   190,   191,   192,   193,   194,   195,   196
};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
    0,     2,     3,     0,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     2,     2,     1,     1,
    6,     2,     3,     1,     1,     1,     5,     3,     0,     1,
    1,     3,     0,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     2,     0,     6,     3,     0,     1,
    1,     1,     2,     4,     1,     6,     3,     0,     1,     1,
    6,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     2,     1,     1,     1,     1,     2,     4,     1,
    1,     1,     1,     1,     0,     1,     1,     2,     2,     1,
    1,     1,     2,     2,     1,     1,     1,     5,     1,     0,
    1,     1,     1,     2,     0,     3,     0,     5,     4,     1,
    1,     1,     0,     5,     2,     3,     1,     2,     3,     6,
    2,     0,     8,     3,     0,     1,     7,     1,     5,     6,
    3,     0,     1,     2,     7,     2,     0,     2,     4,     4,
    6,     2,     0,     6,     2,     2,     0,     2,     0,     7,
    4,     1,     2,     2,     1,     1,     1,     6,     2,     2,
    3,     0,     2,     2,     2,     5,     2,     2,     2,     5,
    5,     2,     1,     1,     1,     1,     1,     1,     1,     3,
    1,     1,     2,     1,     1,     2,     1,     0,     1,     1,
    1,     4,     6,     1,     0,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     4,     0,     3,     0,
    1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
    2,     3,     0,     1,     4,     3,     5,     6,     3,     5
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = WEBIDLEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == WEBIDLEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use WEBIDLerror or WEBIDLUNDEF. */
#define YYERRCODE WEBIDLUNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if WEBIDLDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YY_LOCATION_PRINT
#  if defined WEBIDLLTYPE_IS_TRIVIAL && WEBIDLLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ ( FILE *yyo, YYLTYPE const * const yylocp )
{
    int res = 0;
    int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
    if ( 0 <= yylocp->first_line )
    {
        res += YYFPRINTF ( yyo, "%d", yylocp->first_line );
        if ( 0 <= yylocp->first_column )
        {
            res += YYFPRINTF ( yyo, ".%d", yylocp->first_column );
        }
    }
    if ( 0 <= yylocp->last_line )
    {
        if ( yylocp->first_line < yylocp->last_line )
        {
            res += YYFPRINTF ( yyo, "-%d", yylocp->last_line );
            if ( 0 <= end_col )
            {
                res += YYFPRINTF ( yyo, ".%d", end_col );
            }
        }
        else if ( 0 <= end_col && yylocp->first_column < end_col )
        {
            res += YYFPRINTF ( yyo, "-%d", end_col );
        }
    }
    return res;
}

#   define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

#  else
#   define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#  endif
# endif /* !defined YY_LOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print ( FILE *yyo,
                        yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp )
{
    FILE *yyoutput = yyo;
    YY_USE ( yyoutput );
    YY_USE ( yylocationp );
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
    YY_USE ( yykind );
    YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print ( FILE *yyo,
                  yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp )
{
    YYFPRINTF ( yyo, "%s %s (",
                yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name ( yykind ) );

    YY_LOCATION_PRINT ( yyo, *yylocationp );
    YYFPRINTF ( yyo, ": " );
    yy_symbol_value_print ( yyo, yykind, yyvaluep, yylocationp );
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
yy_reduce_print ( yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                  int yyrule )
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
                          &yyvsp[ ( yyi + 1 ) - ( yynrhs )],
                          & ( yylsp[ ( yyi + 1 ) - ( yynrhs )] ) );
        YYFPRINTF ( stderr, "\n" );
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !WEBIDLDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !WEBIDLDEBUG */


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


/* Context of a parse error.  */
typedef struct
{
    yy_state_t *yyssp;
    yysymbol_kind_t yytoken;
    YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens ( const yypcontext_t *yyctx,
                             yysymbol_kind_t yyarg[], int yyargn )
{
    /* Actual size of YYARG. */
    int yycount = 0;
    int yyn = yypact[+*yyctx->yyssp];
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
            if ( yycheck[yyx + yyn] == yyx && yyx != YYSYMBOL_YYerror
                 && !yytable_value_is_error ( yytable[yyx + yyn] ) )
            {
                if ( !yyarg )
                {
                    ++yycount;
                }
                else if ( yycount == yyargn )
                {
                    return 0;
                }
                else
                {
                    yyarg[yycount++] = YY_CAST ( yysymbol_kind_t, yyx );
                }
            }
    }
    if ( yyarg && yycount == 0 && 0 < yyargn )
    {
        yyarg[0] = YYSYMBOL_YYEMPTY;
    }
    return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen ( const char *yystr )
{
    YYPTRDIFF_T yylen;
    for ( yylen = 0; yystr[yylen]; yylen++ )
    {
        continue;
    }
    return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
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
# endif
#endif



static int
yy_syntax_error_arguments ( const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn )
{
    /* Actual size of YYARG. */
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
    if ( yyctx->yytoken != YYSYMBOL_YYEMPTY )
    {
        int yyn;
        if ( yyarg )
        {
            yyarg[yycount] = yyctx->yytoken;
        }
        ++yycount;
        yyn = yypcontext_expected_tokens ( yyctx,
                                           yyarg ? yyarg + 1 : yyarg, yyargn - 1 );
        if ( yyn == YYENOMEM )
        {
            return YYENOMEM;
        }
        else
        {
            yycount += yyn;
        }
    }
    return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error ( YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                 const yypcontext_t *yyctx )
{
    enum { YYARGS_MAX = 5 };
    /* Internationalized format string. */
    const char *yyformat = YY_NULLPTR;
    /* Arguments of yyformat: reported tokens (one for the "unexpected",
       one per "expected"). */
    yysymbol_kind_t yyarg[YYARGS_MAX];
    /* Cumulated lengths of YYARG.  */
    YYPTRDIFF_T yysize = 0;

    /* Actual size of YYARG. */
    int yycount = yy_syntax_error_arguments ( yyctx, yyarg, YYARGS_MAX );
    if ( yycount == YYENOMEM )
    {
        return YYENOMEM;
    }

    switch ( yycount )
    {
#define YYCASE_(N, S)                       \
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
#undef YYCASE_
    }

    /* Compute error message size.  Don't count the "%s"s, but reserve
       room for the terminator.  */
    yysize = yystrlen ( yyformat ) - 2 * yycount + 1;
    {
        int yyi;
        for ( yyi = 0; yyi < yycount; ++yyi )
        {
            YYPTRDIFF_T yysize1
                = yysize + yystrlen ( yysymbol_name ( yyarg[yyi] ) );
            if ( yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM )
            {
                yysize = yysize1;
            }
            else
            {
                return YYENOMEM;
            }
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
        return -1;
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
                yyp = yystpcpy ( yyp, yysymbol_name ( yyarg[yyi++] ) );
                yyformat += 2;
            }
            else
            {
                ++yyp;
                ++yyformat;
            }
    }
    return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct ( const char *yymsg,
             yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp )
{
    YY_USE ( yyvaluep );
    YY_USE ( yylocationp );
    if ( !yymsg )
    {
        yymsg = "Deleting";
    }
    YY_SYMBOL_PRINT ( yymsg, yykind, yyvaluep, yylocationp );

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    YY_USE ( yykind );
    YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined WEBIDLLTYPE_IS_TRIVIAL && WEBIDLLTYPE_IS_TRIVIAL
    = { 1, 1, 1, 1 }
# endif
      ;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse ( void )
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

    int yyn;
    /* The return value of yyparse.  */
    int yyresult;
    /* Lookahead symbol kind.  */
    yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
    /* The variables used to return semantic value and location from the
       action routines.  */
    YYSTYPE yyval;
    YYLTYPE yyloc;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    /* Buffer for error messages, and its allocated size.  */
    char yymsgbuf[128];
    char *yymsg = yymsgbuf;
    YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

    /* The number of symbols on the RHS of the reduced rule.
       Keep to zero when no symbol should be popped.  */
    int yylen = 0;

    YYDPRINTF ( ( stderr, "Starting parse\n" ) );

    yychar = WEBIDLEMPTY; /* Cause a token to be read.  */
    yylsp[0] = yylloc;
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
            YYLTYPE *yyls1 = yyls;

            /* Each stack pointer address is followed by the size of the
               data in use in that stack, in bytes.  This used to be a
               conditional around just the two extra args, but that might
               be undefined if yyoverflow is a macro.  */
            yyoverflow ( YY_ ( "memory exhausted" ),
                         &yyss1, yysize * YYSIZEOF ( *yyssp ),
                         &yyvs1, yysize * YYSIZEOF ( *yyvsp ),
                         &yyls1, yysize * YYSIZEOF ( *yylsp ),
                         &yystacksize );
            yyss = yyss1;
            yyvs = yyvs1;
            yyls = yyls1;
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
            YYSTACK_RELOCATE ( yyls_alloc, yyls );
#  undef YYSTACK_RELOCATE
            if ( yyss1 != yyssa )
            {
                YYSTACK_FREE ( yyss1 );
            }
        }
# endif

        yyssp = yyss + yysize - 1;
        yyvsp = yyvs + yysize - 1;
        yylsp = yyls + yysize - 1;

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
    if ( yychar == WEBIDLEMPTY )
    {
        YYDPRINTF ( ( stderr, "Reading a token\n" ) );
        yychar = yylex ();
    }

    if ( yychar <= WEBIDLEOF )
    {
        yychar = WEBIDLEOF;
        yytoken = YYSYMBOL_YYEOF;
        YYDPRINTF ( ( stderr, "Now at end of input.\n" ) );
    }
    else if ( yychar == WEBIDLerror )
    {
        /* The scanner already issued an error message, process directly
           to error recovery.  But do not keep the error token as
           lookahead, it is too special and may lead us to an endless
           loop in error recovery. */
        yychar = WEBIDLUNDEF;
        yytoken = YYSYMBOL_YYerror;
        yyerror_range[1] = yylloc;
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
    *++yylsp = yylloc;

    /* Discard the shifted token.  */
    yychar = WEBIDLEMPTY;
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

    /* Default location. */
    YYLLOC_DEFAULT ( yyloc, ( yylsp - yylen ), yylen );
    yyerror_range[1] = yyloc;
    YY_REDUCE_PRINT ( yyn );
    switch ( yyn )
    {

#line 2061 "C:/Code/AeonGUI/mingw64/webidl/parser/webidl_parser.cpp"

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
    *++yylsp = yyloc;

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
    yytoken = yychar == WEBIDLEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE ( yychar );
    /* If not already recovering from an error, report this error.  */
    if ( !yyerrstatus )
    {
        ++yynerrs;
        {
            yypcontext_t yyctx
                = {yyssp, yytoken, &yylloc};
            char const *yymsgp = YY_ ( "syntax error" );
            int yysyntax_error_status;
            yysyntax_error_status = yysyntax_error ( &yymsg_alloc, &yymsg, &yyctx );
            if ( yysyntax_error_status == 0 )
            {
                yymsgp = yymsg;
            }
            else if ( yysyntax_error_status == -1 )
            {
                if ( yymsg != yymsgbuf )
                {
                    YYSTACK_FREE ( yymsg );
                }
                yymsg = YY_CAST ( char *,
                                  YYSTACK_ALLOC ( YY_CAST ( YYSIZE_T, yymsg_alloc ) ) );
                if ( yymsg )
                {
                    yysyntax_error_status
                        = yysyntax_error ( &yymsg_alloc, &yymsg, &yyctx );
                    yymsgp = yymsg;
                }
                else
                {
                    yymsg = yymsgbuf;
                    yymsg_alloc = sizeof yymsgbuf;
                    yysyntax_error_status = YYENOMEM;
                }
            }
            yyerror ( yymsgp );
            if ( yysyntax_error_status == YYENOMEM )
            {
                goto yyexhaustedlab;
            }
        }
    }

    yyerror_range[1] = yylloc;
    if ( yyerrstatus == 3 )
    {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        if ( yychar <= WEBIDLEOF )
        {
            /* Return failure if at end of input.  */
            if ( yychar == WEBIDLEOF )
            {
                YYABORT;
            }
        }
        else
        {
            yydestruct ( "Error: discarding",
                         yytoken, &yylval, &yylloc );
            yychar = WEBIDLEMPTY;
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

        yyerror_range[1] = *yylsp;
        yydestruct ( "Error: popping",
                     YY_ACCESSING_SYMBOL ( yystate ), yyvsp, yylsp );
        YYPOPSTACK ( 1 );
        yystate = *yyssp;
        YY_STACK_PRINT ( yyss, yyssp );
    }

    YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
    *++yyvsp = yylval;
    YY_IGNORE_MAYBE_UNINITIALIZED_END

    yyerror_range[2] = yylloc;
    ++yylsp;
    YYLLOC_DEFAULT ( *yylsp, yyerror_range, 2 );

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


#if 1
    /*-------------------------------------------------.
    | yyexhaustedlab -- memory exhaustion comes here.  |
    `-------------------------------------------------*/
yyexhaustedlab:
    yyerror ( YY_ ( "memory exhausted" ) );
    yyresult = 2;
    goto yyreturn;
#endif


    /*-------------------------------------------------------.
    | yyreturn -- parsing is finished, clean up and return.  |
    `-------------------------------------------------------*/
yyreturn:
    if ( yychar != WEBIDLEMPTY )
    {
        /* Make sure we have latest lookahead translation.  See comments at
           user semantic actions for why this is necessary.  */
        yytoken = YYTRANSLATE ( yychar );
        yydestruct ( "Cleanup: discarding lookahead",
                     yytoken, &yylval, &yylloc );
    }
    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    YYPOPSTACK ( yylen );
    YY_STACK_PRINT ( yyss, yyssp );
    while ( yyssp != yyss )
    {
        yydestruct ( "Cleanup: popping",
                     YY_ACCESSING_SYMBOL ( +*yyssp ), yyvsp, yylsp );
        YYPOPSTACK ( 1 );
    }
#ifndef yyoverflow
    if ( yyss != yyssa )
    {
        YYSTACK_FREE ( yyss );
    }
#endif
    if ( yymsg != yymsgbuf )
    {
        YYSTACK_FREE ( yymsg );
    }
    return yyresult;
}

#line 651 "C:/Code/AeonGUI/webidl/parser/webidl.ypp"

