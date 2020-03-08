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
        }
    }

    static void AddNodes ( Document* aDocument, xmlNode * aNode )
    {
        for ( xmlNode * node = aNode; node; node = node->next )
        {
            if ( node->type == XML_ELEMENT_NODE )
            {
                xmlElementPtr element = reinterpret_cast<xmlElementPtr> ( node );
                AddNodes ( aDocument->AddNode ( Construct ( reinterpret_cast<const char*> ( element->name ), ExtractElementAttributes ( element ) ) ), node->children );
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
        ///@todo use doc->children instead?
        AddNodes ( this, xmlDocGetRootElement ( document ) );
        xmlFreeDoc ( document );
    }

    Document::~Document() = default;

    Node* Document::AddNode ( std::unique_ptr<Node> aNode )
    {
        return mChildren.emplace_back ( std::move ( aNode ) ).get();
    }

    std::unique_ptr<Node> Document::RemoveNode ( const Node* aNode )
    {
        std::unique_ptr<Node> result{};
        auto i = std::find_if ( mChildren.begin(), mChildren.end(), [aNode] ( const std::unique_ptr<Node>& Node )
        {
            return aNode == Node.get();
        } );
        if ( i != mChildren.end() )
        {
            result = std::move ( *i );
            mChildren.erase ( std::remove ( i, mChildren.end(), *i ), mChildren.end() );
        }
        return result;
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aAction )
    {
        for ( auto & mRootNode : mChildren )
        {
            mRootNode->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aAction ) const
    {
        for ( const auto& mRootNode : mChildren )
        {
            static_cast<const Node*> ( mRootNode.get() )->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPostOrder ( const std::function<void ( Node& ) >& aAction )
    {
        for ( auto & mRootNode : mChildren )
        {
            mRootNode->TraverseDepthFirstPostOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPostOrder ( const std::function<void ( const Node& ) >& aAction ) const
    {
        for ( const auto& mRootNode : mChildren )
        {
            static_cast<const Node*> ( mRootNode.get() )->TraverseDepthFirstPostOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble )
    {
        for ( auto & mRootNode : mChildren )
        {
            mRootNode->TraverseDepthFirstPreOrder ( aPreamble, aPostamble );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble ) const
    {
        for ( const auto& mRootNode : mChildren )
        {
            static_cast<const Node*> ( mRootNode.get() )->TraverseDepthFirstPreOrder ( aPreamble, aPostamble );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble, const std::function<bool ( Node& ) >& aUnaryPredicate )
    {
        for ( auto & mRootNode : mChildren )
        {
            mRootNode->TraverseDepthFirstPreOrder ( aPreamble, aPostamble, aUnaryPredicate );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble, const std::function<bool ( const Node& ) >& aUnaryPredicate ) const
    {
        for ( const auto& mRootNode : mChildren )
        {
            static_cast<const Node*> ( mRootNode.get() )->TraverseDepthFirstPreOrder ( aPreamble, aPostamble, aUnaryPredicate );
        }
    }

    void Document::Draw ( Canvas& aCanvas ) const
    {
        TraverseDepthFirstPreOrder (
            [&aCanvas] ( const Node & aNode )
        {
            aNode.DrawStart ( aCanvas );
        },
        [&aCanvas] ( const Node & aNode )
        {
            aNode.DrawFinish ( aCanvas );
        },
        [] ( const Node & aNode )
        {
            return aNode.IsDrawEnabled();
        } );
    }
    void Document::Load ( JavaScript& aJavaScript )
    {
        TraverseDepthFirstPreOrder (
            [&aJavaScript] ( Node & aNode )
        {
            aNode.Load ( aJavaScript );
        } );
    }
    void Document::Unload ( JavaScript& aJavaScript )
    {
        TraverseDepthFirstPreOrder (
            [&aJavaScript] ( Node & aNode )
        {
            aNode.Unload ( aJavaScript );
        } );
    }
}
