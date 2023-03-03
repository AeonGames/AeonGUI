/*
Copyright (C) 2010-2013,2019,2020,2023 Rodrigo Jose Hernandez Cordoba

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
#include <string>
#include "Element.h"
#include "aeongui/Color.h"

namespace AeonGUI
{
    int ParseStyle ( AttributeMap& aAttributeMap, const char* s );
    Element::Element ( const std::string& aTagName, const AttributeMap& aAttributes ) : mTagName{aTagName}, mAttributeMap{aAttributes}
    {
        auto style = mAttributeMap.find ( "style" );
        if ( style != mAttributeMap.end() )
        {
            if ( ParseStyle ( mAttributeMap, std::get<std::string> ( style->second ).c_str() ) )
            {
                auto id = mAttributeMap.find ( "id" );
                if ( id != mAttributeMap.end() )
                {
                    std::cerr << "In Element id = " << std::get<std::string> ( id->second ) << std::endl;
                }
                std::cerr << "Error parsing style: " << std::get<std::string> ( style->second ) << std::endl;
            }
        }
    }

    Element::~Element() = default;

    AttributeType Element::GetAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        auto i = mAttributeMap.find ( attrName );
        if ( i != mAttributeMap.end() )
        {
            return i->second;
        }
        return aDefault;
    }

    void Element::SetAttribute ( const char* attrName, const AttributeType& aValue )
    {
        mAttributeMap[attrName] = aValue;
    }

    AttributeType Element::GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        AttributeType attr = GetAttribute ( attrName );
        Node* parent = parentNode();
        while ( std::holds_alternative<std::monostate> ( attr ) && parent != nullptr )
        {
            if ( parent->nodeType() == ELEMENT_NODE )
            {
                attr = reinterpret_cast<Element*> ( parent )->GetAttribute ( attrName );
            }
            parent = parent->parentNode();
        }
        return std::holds_alternative<std::monostate> ( attr ) ? aDefault : attr;
    }

    Node::NodeType Element::nodeType() const
    {
        return ELEMENT_NODE;
    }
    const std::string& Element::tagName() const
    {
        return mTagName;
    }
}
