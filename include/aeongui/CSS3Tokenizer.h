/*
Copyright (C) 2023 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_CSS3TOKENIZER_H
#define AEONGUI_CSS3TOKENIZER_H
#include <cstdint>
#include <string>
#include "aeongui/Platform.h"

namespace AeonGUI
{
    enum class CSS3TokenType : uint32_t
    {
        IDENT,
        FUNCTION,
        AT_KEYWORD,
        HASH,
        STRING,
        BAD_STRING,
        URL,
        BAD_URL,
        DELIM,
        NUMBER,
        PERCENTAGE,
        DIMENSION,
        WHITESPACE,
        CDO,
        CDC,
        COLON,
        SEMICOLON, COMMA,
        SQUARE_BRACKET_START,
        SQUARE_BRACKET_END,
        PARENTHESES_START,
        PARENTHESES_END,
        CURLY_BRACKET_START,
        CURLY_BRACKET_END,
        END_OF_FILE,
    };

    class CSS3Token
    {
    public:
        DLL CSS3Token ( CSS3TokenType id );
        DLL CSS3TokenType GetType() const;
    private:
        CSS3TokenType mId{};
    };

    class CSS3Tokenizer
    {
    public:
        DLL CSS3Tokenizer();
        DLL static uint32_t GetCodePointFromUT8 ( const char* bytes, uint32_t byte_count, size_t* code_point_lenght = nullptr );
        DLL static CSS3Token Consume ( const std::string& code_points, size_t& pos );
    private:
    };
}
#endif
