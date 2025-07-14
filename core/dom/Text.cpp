/*
Copyright (C) 2020,2024,2025 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/Text.hpp"

namespace AeonGUI
{
    Text::Text ( const std::string& aText, Node* aParent ) : Node{aParent}, mText{aText} {}
    Text::~Text() = default;

    Node::NodeType Text::nodeType() const
    {
        return TEXT_NODE;
    }
    std::string Text::wholeText() const
    {
        auto parent  = parentNode();
        if ( parent != nullptr )
        {
            size_t capacity{sizeof ( std::string::value_type ) }; // initialize with accomodation for the \0 character
            size_t node_count{0};
            // Calculate required capacity
            for ( auto& i : parent->childNodes() )
            {
                if ( i->nodeType() == Node::TEXT_NODE )
                {
                    capacity += reinterpret_cast<const Text*> ( i )->mText.size();
                    node_count++;
                }
            }
            if ( node_count < 2 )
            {
                return mText;
            }
            std::string result{""};
            result.reserve ( capacity );
            for ( auto& i : parent->childNodes() )
            {
                if ( i->nodeType() == Node::TEXT_NODE )
                {
                    result += reinterpret_cast<const Text*> ( i )->mText;
                }
            }
            return result;
        }
        return mText;
    }
}
