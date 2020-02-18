/******************************************************************************
Copyright (C) 2010-2013,2019,2020 Rodrigo Hernandez Cordoba

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
#include <string>
#include <libxml/tree.h>
#include "aeongui/Element.h"
#include "aeongui/Color.h"

namespace AeonGUI
{
    static const std::regex number{"-?([0-9]+|[0-9]*\\.[0-9]+([eE][-+]?[0-9]+)?)"};
    int ParseStyle ( AttributeMap& aAttributeMap, const char* s );
    Element::Element ( xmlElementPtr aXmlElementPtr ) : mXmlElementPtr{aXmlElementPtr}
    {
        if ( mXmlElementPtr == nullptr )
        {
            throw std::runtime_error ( "XML Element is NULL" );
        }
        for ( xmlNodePtr attribute = reinterpret_cast<xmlNodePtr> ( mXmlElementPtr->attributes ); attribute; attribute = attribute->next )
        {
            std::cmatch match;
            const char* value = reinterpret_cast<const char*> ( xmlGetProp ( reinterpret_cast<xmlNodePtr> ( mXmlElementPtr ), attribute->name ) );
            if ( std::regex_match ( value, match, number ) )
            {
                mAttributeMap[reinterpret_cast<const char*> ( attribute->name )] = std::stod ( match[0].str() );
            }
            else if ( std::regex_match ( value, match, Color::ColorRegex ) )
            {
                mAttributeMap[reinterpret_cast<const char*> ( attribute->name )] = Color{match[0].str() };
            }
            else
            {
                mAttributeMap[reinterpret_cast<const char*> ( attribute->name )] = value;
            }
        }
        if ( HasAttr ( "style" ) )
        {
            if ( int error = ParseStyle ( mAttributeMap, GetAttr ( "style" ) ) )
            {
                std::cerr << error << std::endl;
            }
        }
    }

    Element::~Element() = default;

    const char* Element::GetTagName() const
    {
        return reinterpret_cast<const char*> ( mXmlElementPtr->name );
    }

    bool Element::HasAttr ( const char* aAttrName ) const
    {
        return xmlHasProp ( reinterpret_cast<xmlNodePtr> ( mXmlElementPtr ), reinterpret_cast<const xmlChar*> ( aAttrName ) );
    }

    const char* Element::GetAttr ( const char* aAttrName ) const
    {
        return reinterpret_cast<const char*> ( xmlGetProp ( reinterpret_cast<xmlNodePtr> ( mXmlElementPtr ), reinterpret_cast<const xmlChar*> ( aAttrName ) ) );
    }

    double Element::GetAttrAsDouble ( const char* aAttrName, double aDefault ) const
    {
        return ( HasAttr ( aAttrName ) ) ? std::stod ( GetAttr ( aAttrName ) ) : aDefault;
    }

    const char* Element::GetContent () const
    {
        return reinterpret_cast<const char*> ( xmlNodeGetContent ( reinterpret_cast<xmlNodePtr> ( mXmlElementPtr ) ) );
    }

    AttributeType Element::GetAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        auto i = mAttributeMap.find ( attrName );
        if ( i != mAttributeMap.end() )
        {
            return i->second;
        }
        return aDefault;
    }

    AttributeType Element::GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        AttributeType attr = GetAttribute ( attrName );
        Element* parent = mParent;
        while ( std::holds_alternative<std::monostate> ( attr ) && parent != nullptr )
        {
            attr = parent->GetAttribute ( attrName );
            parent = parent->mParent;
        }
        return std::holds_alternative<std::monostate> ( attr ) ? aDefault : attr;
    }

    void Element::DrawStart ( Canvas& aCanvas ) const
    {
        // Do nothing by default
        ( void ) aCanvas;
    }

    void Element::DrawFinish ( Canvas& aCanvas ) const
    {
        // Do nothing by default
        ( void ) aCanvas;
    }

    void Element::Load ( JavaScript& aJavaScript )
    {
        // Do nothing by default
        ( void ) aJavaScript;
    }

    void Element::Unload ( JavaScript& aJavaScript )
    {
        // Do nothing by default
        ( void ) aJavaScript;
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


#define TraverseDepthFirstPostOrder(...) \
    void Element::TraverseDepthFirstPreOrder ( \
        const std::function<void ( __VA_ARGS__ Element& ) >& aPreamble, \
        const std::function<void ( __VA_ARGS__ Element& ) >& aPostamble ) __VA_ARGS__ \
    { \
        auto node = this; \
        aPreamble ( *node ); \
        auto parent = mParent; \
        while ( node != parent ) \
        { \
            if ( node->mIterator < node->mChildren.size() ) \
            { \
                auto prev = node; \
                node = node->mChildren[node->mIterator].get(); \
                aPreamble ( *node ); \
                ++prev->mIterator; \
            } \
            else \
            { \
                aPostamble ( *node ); \
                node->mIterator = 0; \
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
