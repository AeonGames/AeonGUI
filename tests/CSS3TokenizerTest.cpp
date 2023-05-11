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

#include <iostream>
#include <cstdint>
#include <functional>
#include <string>
#include "aeongui/CSS3Tokenizer.h"
#include "gtest/gtest.h"

using namespace ::testing;
namespace AeonGUI
{
    TEST ( CSS3Tokenizer, GetCodePointFromUT8 )
    {
        std::string OneByte{"A"}; // You know "A" right?
        std::string TwoByte{"\xc3\xb1"}; // Spanish Enne (n with ~)
        std::string ThreeByte{"\xe2\x9b\xbd"}; // Fuel Pump
        std::string FourByte{"\xf0\x9f\x92\x80"}; // Skull
        size_t cpl{};
        EXPECT_EQ ( CSS3Tokenizer::GetCodePointFromUT8 ( OneByte.data(), OneByte.size(), &cpl ), 0x41 );
        EXPECT_EQ ( cpl, 1 );
        EXPECT_EQ ( CSS3Tokenizer::GetCodePointFromUT8 ( TwoByte.data(), TwoByte.size(), &cpl ), 0x00F1 );
        EXPECT_EQ ( cpl, 2 );
        EXPECT_EQ ( CSS3Tokenizer::GetCodePointFromUT8 ( ThreeByte.data(), ThreeByte.size(), &cpl ), 0x0026FD );
        EXPECT_EQ ( cpl, 3 );
        EXPECT_EQ ( CSS3Tokenizer::GetCodePointFromUT8 ( FourByte.data(), FourByte.size(), &cpl ), 0x0001F480 );
        EXPECT_EQ ( cpl, 4 );
    }

    TEST ( CSS3Tokenizer, ConsumeComment )
    {
        std::string comment{"/*\n\t\xf0\x9f\x92\x80THIS IS A COMMENT\xf0\x9f\x92\x80\n*/"};
        size_t pos{};
        CSS3Token token{CSS3Tokenizer::Consume ( comment, pos ) };
        EXPECT_EQ ( pos, 32 );
        EXPECT_EQ ( token.GetType(), CSS3TokenType::END_OF_FILE );
    }

    TEST ( CSS3Tokenizer, ConsumeWhitespace )
    {
        std::string whitespace{"\t   \n \n \t     \n/* THIS IS A COMMENT */\n     \t\t\t\t     \n"};
        size_t pos{};
        CSS3Token token{CSS3Tokenizer::Consume ( whitespace, pos ) };
        EXPECT_EQ ( pos, 15 );
        EXPECT_EQ ( token.GetType(), CSS3TokenType::WHITESPACE );
        token = CSS3Tokenizer::Consume ( whitespace, pos );
        EXPECT_EQ ( pos, 54 );
        EXPECT_EQ ( token.GetType(), CSS3TokenType::WHITESPACE );
    }
}
