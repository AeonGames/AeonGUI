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
/* Substitute the variable and function names.  */
#define yyparse         webidlparse
#define yylex           webidllex
#define yyerror         webidlerror
#define yydebug         webidldebug
#define yynerrs         webidlnerrs
#define yylval          webidllval
#define yychar          webidlchar

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

extern "C"
{
    int webidlerror ( const char *s );
}


#line 97 "C:/Code/AeonGUI/mingw64/webidl/parser/webidl_parser.cpp"

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
    YYSYMBOL_ArgumentRest = 135,             /* ArgumentRest  */
    YYSYMBOL_ArgumentName = 136,             /* ArgumentName  */
    YYSYMBOL_Ellipsis = 137,                 /* Ellipsis  */
    YYSYMBOL_Constructor = 138,              /* Constructor  */
    YYSYMBOL_Stringifier = 139,              /* Stringifier  */
    YYSYMBOL_StringifierRest = 140,          /* StringifierRest  */
    YYSYMBOL_StaticMember = 141,             /* StaticMember  */
    YYSYMBOL_StaticMemberRest = 142,         /* StaticMemberRest  */
    YYSYMBOL_Iterable = 143,                 /* Iterable  */
    YYSYMBOL_OptionalType = 144,             /* OptionalType  */
    YYSYMBOL_AsyncIterable = 145,            /* AsyncIterable  */
    YYSYMBOL_OptionalArgumentList = 146,     /* OptionalArgumentList  */
    YYSYMBOL_ReadWriteMaplike = 147,         /* ReadWriteMaplike  */
    YYSYMBOL_MaplikeRest = 148,              /* MaplikeRest  */
    YYSYMBOL_ReadWriteSetlike = 149,         /* ReadWriteSetlike  */
    YYSYMBOL_SetlikeRest = 150,              /* SetlikeRest  */
    YYSYMBOL_Namespace = 151,                /* Namespace  */
    YYSYMBOL_NamespaceMembers = 152,         /* NamespaceMembers  */
    YYSYMBOL_NamespaceMember = 153,          /* NamespaceMember  */
    YYSYMBOL_Dictionary = 154,               /* Dictionary  */
    YYSYMBOL_DictionaryMembers = 155,        /* DictionaryMembers  */
    YYSYMBOL_DictionaryMember = 156,         /* DictionaryMember  */
    YYSYMBOL_DictionaryMemberRest = 157,     /* DictionaryMemberRest  */
    YYSYMBOL_PartialDictionary = 158,        /* PartialDictionary  */
    YYSYMBOL_Default = 159,                  /* Default  */
    YYSYMBOL_Enum = 160,                     /* Enum  */
    YYSYMBOL_EnumValueList = 161,            /* EnumValueList  */
    YYSYMBOL_EnumValueListComma = 162,       /* EnumValueListComma  */
    YYSYMBOL_EnumValueListString = 163,      /* EnumValueListString  */
    YYSYMBOL_CallbackRest = 164,             /* CallbackRest  */
    YYSYMBOL_Typedef = 165,                  /* Typedef  */
    YYSYMBOL_Type = 166,                     /* Type  */
    YYSYMBOL_TypeWithExtendedAttributes = 167, /* TypeWithExtendedAttributes  */
    YYSYMBOL_SingleType = 168,               /* SingleType  */
    YYSYMBOL_UnionType = 169,                /* UnionType  */
    YYSYMBOL_UnionMemberType = 170,          /* UnionMemberType  */
    YYSYMBOL_UnionMemberTypes = 171,         /* UnionMemberTypes  */
    YYSYMBOL_DistinguishableType = 172,      /* DistinguishableType  */
    YYSYMBOL_PrimitiveType = 173,            /* PrimitiveType  */
    YYSYMBOL_UnrestrictedFloatType = 174,    /* UnrestrictedFloatType  */
    YYSYMBOL_FloatType = 175,                /* FloatType  */
    YYSYMBOL_UnsignedIntegerType = 176,      /* UnsignedIntegerType  */
    YYSYMBOL_IntegerType = 177,              /* IntegerType  */
    YYSYMBOL_OptionalLong = 178,             /* OptionalLong  */
    YYSYMBOL_StringType = 179,               /* StringType  */
    YYSYMBOL_PromiseType = 180,              /* PromiseType  */
    YYSYMBOL_RecordType = 181,               /* RecordType  */
    YYSYMBOL_Null = 182,                     /* Null  */
    YYSYMBOL_BufferRelatedType = 183,        /* BufferRelatedType  */
    YYSYMBOL_ExtendedAttributeList = 184,    /* ExtendedAttributeList  */
    YYSYMBOL_ExtendedAttributes = 185,       /* ExtendedAttributes  */
    YYSYMBOL_StringLiteral = 186,            /* StringLiteral  */
    YYSYMBOL_StringLiteralList = 187,        /* StringLiteralList  */
    YYSYMBOL_ExtendedAttribute = 188,        /* ExtendedAttribute  */
    YYSYMBOL_IdentifierList = 189,           /* IdentifierList  */
    YYSYMBOL_Identifiers = 190,              /* Identifiers  */
    YYSYMBOL_ExtendedAttributeNoArgs = 191,  /* ExtendedAttributeNoArgs  */
    YYSYMBOL_ExtendedAttributeArgList = 192, /* ExtendedAttributeArgList  */
    YYSYMBOL_ExtendedAttributeIdent = 193,   /* ExtendedAttributeIdent  */
    YYSYMBOL_ExtendedAttributeIdentList = 194, /* ExtendedAttributeIdentList  */
    YYSYMBOL_ExtendedAttributeNamedArgList = 195, /* ExtendedAttributeNamedArgList  */
    YYSYMBOL_ExtendedAttributeStringLiteral = 196, /* ExtendedAttributeStringLiteral  */
    YYSYMBOL_ExtendedAttributeStringLiteralList = 197 /* ExtendedAttributeStringLiteralList  */
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
         || (defined WEBIDLSTYPE_IS_TRIVIAL && WEBIDLSTYPE_IS_TRIVIAL)))

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
#define YYFINAL  13
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   532

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  87
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  111
/* YYNRULES -- Number of rules.  */
#define YYNRULES  236
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  433

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
    0,   105,   105,   105,   109,   110,   111,   112,   113,   114,
    115,   119,   120,   121,   122,   123,   124,   125,   126,   127,
    128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
    138,   139,   140,   141,   142,   143,   147,   152,   153,   156,
    160,   163,   165,   169,   170,   174,   178,   178,   182,   183,
    187,   187,   191,   192,   193,   194,   195,   196,   197,   198,
    199,   200,   201,   205,   205,   209,   213,   213,   217,   218,
    219,   220,   224,   228,   229,   233,   233,   237,   238,   242,
    246,   247,   248,   252,   253,   257,   258,   259,   260,   264,
    265,   269,   273,   274,   275,   279,   283,   287,   291,   292,
    296,   297,   301,   301,   305,   306,   307,   313,   314,   318,
    322,   326,   327,   328,   332,   336,   336,   340,   341,   345,
    349,   349,   353,   353,   357,   361,   366,   367,   371,   371,
    375,   379,   383,   385,   389,   393,   398,   402,   402,   406,
    410,   410,   414,   417,   421,   424,   427,   430,   430,   434,
    435,   439,   443,   443,   447,   451,   456,   460,   460,   464,
    468,   472,   472,   476,   476,   480,   484,   488,   489,   493,
    497,   498,   499,   503,   507,   508,   512,   512,   516,   529,
    530,   531,   532,   533,   534,   535,   539,   544,   545,   549,
    550,   554,   555,   559,   559,   563,   564,   565,   569,   573,
    577,   577,   581,   582,   583,   584,   585,   586,   587,   588,
    589,   590,   591,   595,   595,   599,   599,   603,   607,   608,
    613,   614,   615,   616,   617,   618,   619,   623,   627,   627,
    631,   635,   639,   643,   647,   651,   655
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if WEBIDLDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name ( yysymbol_kind_t yysymbol ) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
    "\"end of file\"", "error", "\"invalid token\"", "integer", "decimal",
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
    "ArgumentRest", "ArgumentName", "Ellipsis", "Constructor", "Stringifier",
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

#define YYPACT_NINF (-217)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-215)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    11,    47,    34,   103,   -35,   -34,  -217,  -217,  -217,  -217,
    -217,  -217,  -217,  -217,    62,    19,    77,    80,    82,    44,
    15,    11,  -217,  -217,  -217,  -217,  -217,  -217,  -217,    17,
    -18,    47,    37,    98,    31,   102,   101,  -217,    49,    55,
    59,    22,  -217,  -217,   131,   426,  -217,    56,  -217,    67,
    -217,    57,    64,    95,   -34,  -217,    66,   426,    75,    24,
    135,    78,   144,   -38,    79,   151,   145,  -217,  -217,    85,
    89,    76,  -217,  -217,  -217,  -217,  -217,  -217,   108,    83,
    -217,   -22,  -217,  -217,    90,  -217,    90,  -217,  -217,  -217,
    -217,   -18,   100,   106,    84,    86,  -217,    15,  -217,    15,
    -217,  -217,  -217,   104,    20,    49,  -217,  -217,  -217,  -217,
    -14,   107,   109,   119,   233,    21,   121,   192,  -217,  -217,
    -217,  -217,    89,  -217,  -217,  -217,   426,    90,   164,   126,
    -217,  -217,    40,   120,   195,  -217,   197,  -217,  -217,    64,
    499,   -18,   133,   118,   127,   134,   -14,   184,   200,  -217,
    137,   138,   205,  -217,   -38,    30,   141,   394,    26,   143,
    -217,   139,  -217,   -22,  -217,  -217,  -217,  -217,    90,  -217,
    100,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,
    -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,
    -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,
    142,   140,   152,    23,    20,  -217,  -217,    27,   153,  -217,
    15,  -217,   107,  -217,  -217,  -217,    15,  -217,  -217,  -217,
    -217,  -217,   157,  -217,  -217,   165,   219,  -217,  -217,   205,
    158,   159,    16,   160,  -217,   220,   -16,    21,  -217,  -217,
    -217,  -217,  -217,  -217,  -217,  -217,   426,  -217,  -217,  -217,
    -217,  -217,  -217,  -217,  -217,   170,   179,   -14,  -217,   213,
    243,  -217,    13,   426,   174,  -217,  -217,   246,  -217,  -217,
    177,   340,  -217,   248,  -217,     9,   -18,  -217,   171,  -217,
    15,    15,  -217,  -217,  -217,  -217,    15,  -217,   205,  -217,
    -217,   205,  -217,  -217,  -217,   178,    26,  -217,   205,  -217,
    -217,   181,   -22,   176,    90,  -217,  -217,  -217,  -217,  -217,
    -217,  -217,  -217,   182,  -217,  -217,  -217,  -217,   253,  -217,
    186,   191,   188,    27,  -217,  -217,   194,  -217,  -217,  -217,
    198,  -217,   189,    15,   190,   196,   193,   426,   426,  -217,
    -217,  -217,   199,   213,  -217,   234,   207,  -217,   499,    29,
    -217,   -18,  -217,   426,  -217,   206,   190,    15,   208,    15,
    215,  -217,  -217,  -217,  -217,   210,   217,  -217,   221,   216,
    290,  -217,   214,  -217,   224,   223,  -217,    15,   294,  -217,
    228,   142,   229,  -217,   236,   231,  -217,  -217,   241,   -18,
    242,  -217,    90,  -217,   227,  -217,   258,  -217,    90,   271,
    90,   425,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,
    -217,  -217,  -217,    90,   272,   240,    15,   235,    90,   275,
    245,    15,   247,    90,   268,   250,    90,    40,  -217,   252,
    15,   254,  -217
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
    214,     0,     0,     0,   230,   216,   220,   221,   222,   223,
    224,   225,   226,     1,     0,     0,     0,     0,     0,     0,
    214,   214,     4,     6,    10,     5,     7,     8,     9,     0,
    121,     0,     0,     0,     0,     0,     0,    73,    64,     0,
    0,     0,    40,    42,     0,     0,     2,   232,   217,     0,
    235,     0,   123,     0,   216,   213,     0,     0,     0,     0,
    0,     0,     0,   214,     0,     0,     0,    43,    44,     0,
    0,     0,   171,   182,   183,   184,   185,   191,   194,     0,
    181,   214,   169,   167,   201,   170,   201,   180,   179,   190,
    172,   121,   229,   219,     0,     0,   231,   214,   120,   214,
    124,   215,    72,     0,   214,    64,    36,    37,    38,    63,
    153,   162,     0,     0,     0,   214,     0,     0,    41,   166,
    187,   188,     0,   189,   193,   192,     0,   201,     0,     0,
    200,   168,     0,     0,     0,   227,     0,   236,   233,   123,
    0,   121,     0,     0,     0,     0,   153,     0,   164,   160,
    0,     0,     0,   149,   214,   116,     0,     0,   214,     0,
    186,     0,   175,   214,   174,   195,   196,   197,   201,   234,
    229,   218,   122,   127,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    28,    29,    30,    31,    32,    33,    34,    35,   126,
    158,     0,     0,     0,   214,    77,    78,   214,     0,   152,
    214,   154,   162,   161,   159,   146,   214,   150,   147,   118,
    119,   109,     0,   115,   117,     0,     0,   113,   111,     0,
    0,     0,     0,     0,   112,   103,   103,   214,    52,    58,
    59,    62,    95,    53,   107,   108,     0,    54,    55,    56,
    57,    60,   142,    61,   144,     0,   103,   153,   198,   177,
    0,   228,     0,     0,     0,    74,    90,     0,    89,    75,
    0,     0,   151,     0,   163,     0,   121,    45,     0,    96,
    214,   214,    91,    92,    93,    94,   214,   102,     0,   134,
    133,     0,   131,    50,   110,     0,   214,    68,     0,    69,
    70,     0,   214,     0,   201,    82,    85,   105,    83,    84,
    87,    86,    88,     0,   104,    80,    81,   157,   129,   165,
    0,     0,     0,   214,    48,    49,     0,    99,   100,   101,
    0,    98,     0,   214,   138,     0,     0,     0,     0,    65,
    66,    71,     0,   177,   173,     0,     0,   128,     0,     0,
    39,   121,    46,     0,    97,     0,   138,   214,     0,   214,
    0,   135,   132,   156,   176,     0,     0,   125,     0,     0,
    0,   114,     0,   137,     0,     0,   145,   214,     0,    79,
    0,   158,   141,   136,     0,     0,   106,   130,     0,   121,
    0,   143,   201,   155,     0,   139,     0,   140,   201,     0,
    201,     0,   202,   203,   204,   205,   206,   207,   208,   209,
    210,   211,   212,   201,     0,     0,   214,     0,   201,     0,
    0,   214,     0,   201,     0,     0,   201,     0,   178,     0,
    214,     0,   199
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -217,   309,  -217,  -217,  -217,  -217,  -217,  -217,  -217,  -217,
        -217,     8,  -217,   105,    69,   232,   284,    48,  -217,  -217,
        -217,   154,  -217,  -136,    -4,  -217,  -217,  -217,  -217,  -217,
        -217,  -217,  -143,  -217,  -217,  -216,  -217,  -217,  -112,  -217,
        -217,  -217,  -217,  -217,  -217,   -83,   209,   249,  -217,    -1,
        -217,  -217,    99,  -217,  -217,  -217,  -217,     4,  -217,  -217,
        -217,   124,  -217,   132,   344,   211,  -217,  -217,  -134,  -217,
        -217,  -217,    -9,  -217,  -217,   155,  -217,  -217,  -217,   -42,
        -94,  -217,   -75,  -159,    32,   251,   173,  -217,   255,  -217,
        303,  -217,   -49,  -217,  -217,   -85,  -217,     0,   327,   353,
        262,   352,  -217,   218,  -217,  -217,  -217,  -217,  -217,  -217,
        -217
    };

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
    0,     2,    21,   199,    22,   106,   107,    23,    42,    66,
    67,   270,   323,   156,   237,    61,    68,   255,   296,    24,
    36,   142,   204,   238,   314,   315,   316,   267,   239,   282,
    240,   241,   242,   330,   331,   288,   317,   243,   244,   245,
    246,   221,   222,   223,   224,    51,    98,    52,   100,   200,
    348,   325,   247,   292,   248,   289,   249,   358,   250,   390,
    251,   252,   253,   254,    25,   113,   154,    26,   145,   146,
    211,   118,   263,    27,   112,   149,   213,    37,    28,   155,
    44,    83,    84,   128,   303,    85,    86,    87,   122,    88,
    89,   125,   168,    90,   426,   131,   413,    45,    32,    93,
    94,     5,    95,   135,     6,     7,     8,     9,    10,    11,
    12
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
    3,   132,   153,    82,   259,   140,   127,   205,   133,   217,
    287,    -3,   209,  -214,   327,   103,   305,   306,   328,   307,
    291,     3,    47,    48,    34,  -214,   216,    64,   266,   105,
    53,   206,   305,   306,    13,   219,   329,  -148,   231,    35,
    298,     1,   162,    29,   233,    65,    30,    65,   220,    31,
    308,   309,     4,   310,   311,   312,    70,     1,   201,    81,
    290,     1,    71,   114,    41,     1,   308,   309,    18,   310,
    311,   312,    92,    48,    73,    74,    75,    76,    77,    78,
    33,   129,    38,   260,   161,    39,   279,    40,   127,   283,
    1,    53,   313,    80,     1,   -76,   -51,    53,    49,     1,
    1,   -67,   -47,    56,   143,     1,     1,    58,    14,    57,
    147,   165,   166,   167,    15,   157,   273,    55,    16,    17,
    297,    59,   275,   301,   120,   121,    60,    18,    19,    62,
    203,    77,    78,    63,   294,    20,    69,    91,    99,    96,
    109,    53,   102,   343,   299,   337,   147,    97,   338,   104,
    111,    70,   110,   115,   114,   341,   116,    71,   256,    70,
    117,   119,    72,   129,   124,    71,   137,   126,   138,    73,
    74,    75,    76,    77,    78,    79,   130,    73,    74,    75,
    76,    77,    78,   134,   150,   141,   334,   335,    80,   136,
    148,   203,   336,   332,   151,   158,    80,   159,   163,    81,
    170,   207,   169,    48,   143,   287,   212,   271,   202,   208,
    236,   210,    70,   214,   215,   216,   225,   257,    71,   345,
    262,   318,   264,    72,   258,   361,   362,   127,   265,   272,
    73,    74,    75,    76,    77,    78,    79,   157,   276,   356,
    278,   277,   280,   281,   286,   295,   287,   302,   304,    80,
    319,   320,   321,   326,   339,   333,   342,   147,   344,   152,
    81,   347,   346,   373,   349,   375,    70,   350,   369,   351,
    353,   355,    71,   357,   354,   363,    53,    72,   360,   359,
    365,   366,   371,   385,    73,    74,    75,    76,    77,    78,
    79,   376,   378,   374,   377,   381,   256,   379,   380,   382,
    383,   386,   129,    80,   387,   398,   394,   396,   384,   397,
    389,   370,   391,   399,    81,   401,   392,   393,   395,   400,
    418,   415,   417,   271,   416,   420,   425,   422,   414,   421,
    46,   352,   423,   419,   427,   430,   431,   144,   424,   432,
    324,   428,   293,   108,   340,   368,   139,   367,   172,   226,
    216,    53,   203,   322,   227,   300,   284,   228,   269,   229,
    372,   230,   231,    43,   285,   218,   232,   274,   233,   234,
    235,   236,   388,    70,   123,   364,   268,   160,   429,    71,
    164,   101,    50,    54,    72,     0,     0,     0,   261,    53,
    0,    73,    74,    75,    76,    77,    78,    79,   171,     0,
    0,     0,     0,   226,   216,     0,   203,     0,   227,     0,
    80,   228,     0,   229,     0,   230,   231,     0,     0,     0,
    232,    81,   233,   234,   235,   236,     0,    70,     0,     0,
    0,     0,     0,    71,     0,     0,     0,     0,    72,     0,
    0,     0,     0,     0,     0,    73,    74,    75,    76,    77,
    78,    79,     0,     0,     0,     0,     0,     0,     0,    70,
    0,     0,     0,     0,    80,    71,     0,     0,     0,     0,
    72,     0,     0,     0,     0,    81,     0,    73,    74,    75,
    76,    77,    78,    79,   402,   403,   404,   405,   406,   407,
    408,   409,   410,   411,   412,     0,    80,     0,     0,     0,
    0,     0,     0,     0,   173,     0,     0,    81,   174,   175,
    176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
    186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
    196,   197,   198
};

static const yytype_int16 yycheck[] =
{
    0,    86,   114,    45,   163,    99,    81,   143,    91,   152,
    26,     0,   146,    27,     5,    57,     3,     4,     9,     6,
    236,    21,     5,     6,     5,    43,    10,     5,     5,     5,
    30,   143,     3,     4,     0,     5,    27,    75,    22,    20,
    256,    79,   127,    78,    28,    23,    81,    23,    18,    83,
    37,    38,     5,    40,    41,    42,    33,    79,   141,    81,
    76,    79,    39,    63,    20,    79,    37,    38,    24,    40,
    41,    42,     5,     6,    51,    52,    53,    54,    55,    56,
    18,    81,     5,   168,   126,     5,   229,     5,   163,   232,
    79,    91,    79,    70,    79,    75,    75,    97,    81,    79,
    79,    75,    75,     5,   104,    79,    79,     5,     5,    78,
    110,    71,    72,    73,    11,   115,   210,    80,    15,    16,
    256,    20,   216,   257,    35,    36,    77,    24,    25,    74,
    12,    55,    56,    74,   246,    32,     5,    81,    43,    82,
    5,   141,    76,   302,   256,   288,   146,    83,   291,    74,
    6,    33,    74,    74,   154,   298,     5,    39,   158,    33,
    15,    76,    44,   163,    56,    39,    82,    84,    82,    51,
    52,    53,    54,    55,    56,    57,    86,    51,    52,    53,
    54,    55,    56,    83,    75,    81,   280,   281,    70,    83,
    83,    12,   286,   276,    75,    74,    70,     5,    34,    81,
    5,    74,    82,     6,   204,    26,     6,   207,    75,    75,
    31,    27,    33,    76,    76,    10,    75,    74,    39,   304,
    78,   263,    82,    44,    85,   337,   338,   302,    76,    76,
    51,    52,    53,    54,    55,    56,    57,   237,    81,   333,
    21,    76,    84,    84,    84,    75,    26,    34,     5,    70,
    76,     5,    75,     5,    76,    84,    75,   257,    82,    26,
    81,     8,    80,   357,    78,   359,    33,    76,   351,    81,
    76,    82,    39,    83,    76,    76,   276,    44,    85,    83,
    46,    74,    76,   377,    51,    52,    53,    54,    55,    56,
    57,    76,    75,    85,    84,     5,   296,    76,    82,    85,
    76,     7,   302,    70,    76,    47,   389,   392,    85,    82,
    81,   353,    76,   398,    81,   400,    85,    76,    76,    48,
    85,    49,   416,   323,    84,    50,    58,   421,   413,    84,
    21,   323,    85,   418,    84,    83,   430,   105,   423,    85,
    271,   426,   237,    59,   296,   349,    97,   348,   139,     9,
    10,   351,    12,    13,    14,   256,   232,    17,   204,    19,
    356,    21,    22,    19,   232,   154,    26,   212,    28,    29,
    30,    31,   381,    33,    71,   343,   203,   122,   427,    39,
    129,    54,    29,    31,    44,    -1,    -1,    -1,   170,   389,
    -1,    51,    52,    53,    54,    55,    56,    57,   136,    -1,
    -1,    -1,    -1,     9,    10,    -1,    12,    -1,    14,    -1,
    70,    17,    -1,    19,    -1,    21,    22,    -1,    -1,    -1,
    26,    81,    28,    29,    30,    31,    -1,    33,    -1,    -1,
    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,    -1,
    -1,    -1,    -1,    -1,    -1,    51,    52,    53,    54,    55,
    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    33,
    -1,    -1,    -1,    -1,    70,    39,    -1,    -1,    -1,    -1,
    44,    -1,    -1,    -1,    -1,    81,    -1,    51,    52,    53,
    54,    55,    56,    57,    59,    60,    61,    62,    63,    64,
    65,    66,    67,    68,    69,    -1,    70,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,     5,    -1,    -1,    81,     9,    10,
    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
    21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
    31,    32,    33
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
    0,    79,    88,   184,     5,   188,   191,   192,   193,   194,
    195,   196,   197,     0,     5,    11,    15,    16,    24,    25,
    32,    89,    91,    94,   106,   151,   154,   160,   165,    78,
    81,    83,   185,    18,     5,    20,   107,   164,     5,     5,
    5,    20,    95,   151,   167,   184,    88,     5,     6,    81,
    186,   132,   134,   184,   188,    80,     5,    78,     5,    20,
    77,   102,    74,    74,     5,    23,    96,    97,   103,     5,
    33,    39,    44,    51,    52,    53,    54,    55,    56,    57,
    70,    81,   166,   168,   169,   172,   173,   174,   176,   177,
    180,    81,     5,   186,   187,   189,    82,    83,   133,    43,
    135,   185,    76,   166,    74,     5,    92,    93,   103,     5,
    74,     6,   161,   152,   184,    74,     5,    15,   158,    76,
    35,    36,   175,   177,    56,   178,    84,   169,   170,   184,
    86,   182,   182,   132,    83,   190,    83,    82,    82,   134,
    167,    81,   108,   184,   102,   155,   156,   184,    83,   162,
    75,    75,    26,   125,   153,   166,   100,   184,    74,     5,
    175,   166,   182,    34,   172,    71,    72,    73,   179,    82,
    5,   187,   133,     5,     9,    10,    11,    12,    13,    14,
    15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
    25,    26,    27,    28,    29,    30,    31,    32,    33,    90,
    136,   132,    75,    12,   109,   110,   125,    74,    75,   155,
    27,   157,     6,   163,    76,    76,    10,   119,   152,     5,
    18,   128,   129,   130,   131,    75,     9,    14,    17,    19,
    21,    22,    26,    28,    29,    30,    31,   101,   110,   115,
    117,   118,   119,   124,   125,   126,   127,   139,   141,   143,
    145,   147,   148,   149,   150,   104,   184,    74,    85,   170,
    182,   190,    78,   159,    82,    76,     5,   114,   173,   108,
    98,   184,    76,   167,   162,   167,    81,    76,    21,   119,
    84,    84,   116,   119,   148,   150,    84,    26,   122,   142,
    76,   122,   140,   100,   125,    75,   105,   110,   122,   125,
    139,   155,    34,   171,     5,     3,     4,     6,    37,    38,
    40,    41,    42,    79,   111,   112,   113,   123,   166,    76,
    5,    75,    13,    99,   101,   138,     5,     5,     9,    27,
    120,   121,   132,    84,   167,   167,   167,   119,   119,    76,
    104,   119,    75,   170,    82,   182,    80,     8,   137,    78,
    76,    81,    98,    76,    76,    82,   167,    83,   144,    83,
    85,   125,   125,    76,   171,    46,    74,   136,   111,   132,
    166,    76,   144,   167,    85,   167,    76,    84,    75,    76,
    82,     5,    85,    76,    85,   167,     7,    76,   159,    81,
    146,    76,    85,    76,   132,    76,   182,    82,    47,   182,
    48,   182,    59,    60,    61,    62,    63,    64,    65,    66,
    67,    68,    69,   183,   182,    49,    84,   167,    85,   182,
    50,    84,   167,    85,   182,    58,   181,    84,   182,   179,
    83,   167,    85
};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
    0,    87,    88,    88,    89,    89,    89,    89,    89,    89,
    89,    90,    90,    90,    90,    90,    90,    90,    90,    90,
    90,    90,    90,    90,    90,    90,    90,    90,    90,    90,
    90,    90,    90,    90,    90,    90,    91,    92,    92,    93,
    94,    95,    95,    96,    96,    97,    98,    98,    99,    99,
    100,   100,   101,   101,   101,   101,   101,   101,   101,   101,
    101,   101,   101,   102,   102,   103,   104,   104,   105,   105,
    105,   105,   106,   107,   107,   108,   108,   109,   109,   110,
    111,   111,   111,   112,   112,   113,   113,   113,   113,   114,
    114,   115,   116,   116,   116,   117,   118,   119,   120,   120,
    121,   121,   122,   122,   123,   123,   123,   124,   124,   125,
    126,   127,   127,   127,   128,   129,   129,   130,   130,   131,
    132,   132,   133,   133,   134,   135,   136,   136,   137,   137,
    138,   139,   140,   140,   141,   142,   143,   144,   144,   145,
    146,   146,   147,   148,   149,   150,   151,   152,   152,   153,
    153,   154,   155,   155,   156,   157,   158,   159,   159,   160,
    161,   162,   162,   163,   163,   164,   165,   166,   166,   167,
    168,   168,   168,   169,   170,   170,   171,   171,   172,   173,
    173,   173,   173,   173,   173,   173,   174,   175,   175,   176,
    176,   177,   177,   178,   178,   179,   179,   179,   180,   181,
    182,   182,   183,   183,   183,   183,   183,   183,   183,   183,
    183,   183,   183,   184,   184,   185,   185,   186,   187,   187,
    188,   188,   188,   188,   188,   188,   188,   189,   190,   190,
    191,   192,   193,   194,   195,   196,   197
};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
    0,     2,     3,     0,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     1,     1,     1,     4,     1,     1,     6,
    2,     3,     1,     1,     1,     5,     3,     0,     1,     1,
    3,     0,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     2,     0,     6,     3,     0,     1,     1,
    1,     2,     4,     1,     6,     3,     0,     1,     1,     6,
    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     2,     1,     1,     1,     1,     2,     4,     1,     1,
    1,     1,     1,     0,     1,     1,     5,     1,     1,     2,
    2,     1,     1,     1,     5,     1,     0,     1,     1,     1,
    2,     0,     3,     0,     2,     7,     1,     1,     1,     0,
    5,     2,     3,     1,     2,     3,     6,     2,     0,     8,
    3,     0,     1,     7,     1,     5,     6,     3,     0,     1,
    2,     7,     2,     0,     2,     8,     6,     2,     0,     6,
    2,     2,     0,     2,     0,     7,     4,     1,     2,     2,
    1,     1,     1,     6,     2,     2,     3,     0,    29,     1,
    1,     1,     1,     1,     1,     1,     3,     1,     1,     2,
    1,     1,     2,     1,     0,     1,     1,     1,     4,     6,
    1,     0,     1,     1,     1,     1,     1,     1,     1,     1,
    1,     1,     1,     4,     0,     3,     0,     1,     3,     1,
    1,     1,     1,     1,     1,     1,     1,     2,     3,     0,
    1,     4,     3,     5,     6,     3,     5
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
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print ( FILE *yyo,
                        yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep )
{
    FILE *yyoutput = yyo;
    YY_USE ( yyoutput );
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
                  yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep )
{
    YYFPRINTF ( yyo, "%s %s (",
                yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name ( yykind ) );

    yy_symbol_value_print ( yyo, yykind, yyvaluep );
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
                          &yyvsp[ ( yyi + 1 ) - ( yynrhs )] );
        YYFPRINTF ( stderr, "\n" );
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
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






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct ( const char *yymsg,
             yysymbol_kind_t yykind, YYSTYPE *yyvaluep )
{
    YY_USE ( yyvaluep );
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

    int yyn;
    /* The return value of yyparse.  */
    int yyresult;
    /* Lookahead symbol kind.  */
    yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
    /* The variables used to return semantic value and location from the
       action routines.  */
    YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

    /* The number of symbols on the RHS of the reduced rule.
       Keep to zero when no symbol should be popped.  */
    int yylen = 0;

    YYDPRINTF ( ( stderr, "Starting parse\n" ) );

    yychar = WEBIDLEMPTY; /* Cause a token to be read.  */
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


    YY_REDUCE_PRINT ( yyn );
    switch ( yyn )
    {

#line 1658 "C:/Code/AeonGUI/mingw64/webidl/parser/webidl_parser.cpp"

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
    yytoken = yychar == WEBIDLEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE ( yychar );
    /* If not already recovering from an error, report this error.  */
    if ( !yyerrstatus )
    {
        ++yynerrs;
        yyerror ( YY_ ( "syntax error" ) );
    }

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
                         yytoken, &yylval );
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


        yydestruct ( "Error: popping",
                     YY_ACCESSING_SYMBOL ( yystate ), yyvsp );
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
                     yytoken, &yylval );
    }
    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    YYPOPSTACK ( yylen );
    YY_STACK_PRINT ( yyss, yyssp );
    while ( yyssp != yyss )
    {
        yydestruct ( "Cleanup: popping",
                     YY_ACCESSING_SYMBOL ( +*yyssp ), yyvsp );
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

#line 657 "C:/Code/AeonGUI/webidl/parser/webidl.ypp"

extern "C"
{
    int webidlerror ( const char *s )
    {
        std::cerr << s << std::endl;
        return 0;
    }
}
