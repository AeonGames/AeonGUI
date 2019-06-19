/******************************************************************************
Copyright (C) 2010-2013,2019 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include <iostream>
#include <libxml/tree.h>
#include "aeongui/Element.h"

namespace AeonGUI
{
    Element::Element ( xmlElementPtr aXmlElementPtr ) : mXmlElementPtr{aXmlElementPtr}
    {
        if ( mXmlElementPtr == nullptr )
        {
            throw std::runtime_error ( "XML Element is NULL" );
        }
        std::cout << mXmlElementPtr->name << std::endl;
        for ( auto* attribute = mXmlElementPtr->attributes; attribute; attribute = attribute->nexth )
        {
            std::cout << "\t" << attribute->name << " " << xmlGetProp ( reinterpret_cast<xmlNodePtr> ( mXmlElementPtr ), attribute->name ) << std::endl;
        }
    }

    const uint8_t* Element::GetTagName() const
    {
        return mXmlElementPtr->name;
    }

    /*  This is ugly, but it is only way to use the same code for the const and the non const version
        without having to add template or friend members to the class declaration. */
#define TraverseDepthFirstPreOrder(...) \
    void Element::TraverseDepthFirstPreOrder ( const std::function<void ( __VA_ARGS__ Element& ) >& aAction ) __VA_ARGS__ \
    {\
        /** @todo (EC++ Item 3) This code is the same as the constant overload,\
        but can't easily be implemented in terms of that because of aAction's Element parameter\
        need to also be const.\
        */\
        auto Element{this};\
        aAction ( *Element );\
        auto parent = mParent;\
        while ( Element != parent )\
        {\
            if ( Element->mIterator < Element->mChildren.size() )\
            {\
                auto prev = Element;\
                Element = Element->mChildren[Element->mIterator].get();\
                aAction ( *Element );\
                prev->mIterator++;\
            }\
            else\
            {\
                Element->mIterator = 0; /* Reset counter for next traversal.*/\
                Element = Element->mParent;\
            }\
        }\
    }

    TraverseDepthFirstPreOrder ( const )
    TraverseDepthFirstPreOrder( )
#undef TraverseDepthFirstPreOrder

#define TraverseDepthFirstPostOrder(...) \
    void Element::TraverseDepthFirstPostOrder ( const std::function<void ( __VA_ARGS__ Element& ) >& aAction ) __VA_ARGS__ \
    { \
        /* \
        This code implements a similar solution to this stackoverflow answer: \
        http://stackoverflow.com/questions/5987867/traversing-a-n-ary-tree-without-using-recurrsion/5988138#5988138 \
        */ \
        auto node = this; \
        auto parent = mParent; \
        while ( node != parent ) \
        { \
            if ( node->mIterator < node->mChildren.size() ) \
            { \
                auto prev = node; \
                node = node->mChildren[node->mIterator].get(); \
                ++prev->mIterator; \
            } \
            else \
            { \
                aAction ( *node ); \
                node->mIterator = 0; /* Reset counter for next traversal. */ \
                node = node->mParent; \
            } \
        } \
    }

    TraverseDepthFirstPostOrder ( const )
    TraverseDepthFirstPostOrder( )
#undef TraverseDepthFirstPostOrder

    Element* Element::AddElement ( std::unique_ptr<Element> aElement )
    {
        aElement->mParent = this;
        return mChildren.emplace_back ( std::move ( aElement ) ).get();
    }

    std::unique_ptr<Element> Element::RemoveElement ( const Element* aElement )
    {
        std::unique_ptr<Element> result{};
        auto i = std::find_if ( mChildren.begin(), mChildren.end(), [aElement] ( const std::unique_ptr<Element>& Element )
        {
            return aElement == Element.get();
        } );
        if ( i != mChildren.end() )
        {
            result = std::move ( *i );
            mChildren.erase ( std::remove ( i, mChildren.end(), *i ), mChildren.end() );
        }
        result->mParent = nullptr;
        return result;
    }
}
