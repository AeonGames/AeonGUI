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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "aeongui/Document.h"
#include "aeongui/ElementFactory.h"
#include "dom/Text.h"
#include "dom/Element.h"

namespace AeonGUI
{
    static const std::regex number{"-?([0-9]+|[0-9]*\\.[0-9]+([eE][-+]?[0-9]+)?)"};
    static AttributeMap ExtractElementAttributes ( xmlElementPtr aXmlElementPtr )
    {
        AttributeMap attribute_map{};
        for ( xmlNodePtr attribute = reinterpret_cast<xmlNodePtr> ( aXmlElementPtr->attributes ); attribute; attribute = attribute->next )
        {
            std::cmatch match;
            const char* value = reinterpret_cast<const char*> ( xmlGetProp ( reinterpret_cast<xmlNodePtr> ( aXmlElementPtr ), attribute->name ) );
            if ( std::regex_match ( value, match, number ) )
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = std::stod ( match[0].str() );
            }
            else if ( std::regex_match ( value, match, Color::ColorRegex ) )
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = Color{match[0].str() };
            }
            else
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = value;
            }
        }
        return attribute_map;
    }
    static void AddNodes ( Node* aNode, xmlNode * aXmlNode )
    {
        for ( xmlNode * node = aXmlNode; node; node = node->next )
        {
            if ( node->type == XML_ELEMENT_NODE )
            {
                xmlElementPtr element = reinterpret_cast<xmlElementPtr> ( node );
                AddNodes ( aNode->AddNode ( Construct ( reinterpret_cast<const char*> ( element->name ), ExtractElementAttributes ( element ) ) ), node->children );
            }
            else if ( xmlNodeIsText ( node ) && !xmlIsBlankNode ( node ) )
            {
                AddNodes ( aNode->AddNode ( std::make_unique<Text> ( reinterpret_cast<const char*> ( node->content ) ) ), node->children );
            }
        }
    }

    Document::Document () = default;

    Document::Document ( const std::string& aFilename )
    {
        xmlDocPtr document{xmlReadFile ( aFilename.c_str(), nullptr, 0 ) };
        if ( document == nullptr )
        {
            throw std::runtime_error ( "Could not parse xml file" );
        }
        ///@todo use document->children instead?
        xmlElementPtr root_element = reinterpret_cast<xmlElementPtr> ( xmlDocGetRootElement ( document ) );
        mDocumentElement = Construct ( reinterpret_cast<const char*> ( root_element->name ), ExtractElementAttributes ( root_element ) );
        AddNodes ( mDocumentElement.get(), root_element->children );
        xmlFreeDoc ( document );

        // Evaluate all script nodes.
        mDocumentElement->TraverseDepthFirstPreOrder (
            [this] ( Node * aNode )
        {
            if ( aNode->nodeType() == Node::ELEMENT_NODE && reinterpret_cast<Element*> ( aNode )->tagName() == "script" )
            {
                const auto& children = aNode->childNodes();
                ///@todo Do not asume script elements contain elements or more than one text node.
                auto text_node = std::find_if ( children.begin(), children.end(), [] ( const std::unique_ptr<Node>& aChild )
                {
                    return aChild->nodeType() == Node::TEXT_NODE;
                } );
                if ( text_node != children.end() )
                {
                    mJavaScript.Eval ( reinterpret_cast<const Text*> ( text_node->get() )->wholeText() );
                }
            }
        } );
        /**@todo Emit onload event.*/
    }

    Document::~Document() = default;
    Node* Document::documentElement()
    {
        return mDocumentElement.get();
    }

    void Document::Draw ( Canvas& aCanvas ) const
    {
        mDocumentElement->TraverseDepthFirstPreOrder (
            [&aCanvas] ( const Node * aNode )
        {
            aNode->DrawStart ( aCanvas );
        },
        [&aCanvas] ( const Node * aNode )
        {
            aNode->DrawFinish ( aCanvas );
        },
        [] ( const Node * aNode )
        {
            return aNode->IsDrawEnabled();
        } );
    }
}
