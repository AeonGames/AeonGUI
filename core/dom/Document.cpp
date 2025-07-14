/*
Copyright (C) 2019,2020,2023,2024,2025 Rodrigo Jose Hernandez Cordoba

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
#include <libcss/libcss.h>
#include "aeongui/ElementFactory.h"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Text.hpp"
#include "aeongui/dom/Element.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        static const std::regex number{"-?([0-9]+|[0-9]*\\.[0-9]+([eE][-+]?[0-9]+)?)"};
        static AttributeMap ExtractElementAttributes ( xmlElementPtr aXmlElementPtr )
        {
            AttributeMap attribute_map{};
            for ( xmlNodePtr attribute = reinterpret_cast<xmlNodePtr> ( aXmlElementPtr->attributes ); attribute; attribute = attribute->next )
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = reinterpret_cast<const char*> ( xmlGetProp ( reinterpret_cast<xmlNodePtr> ( aXmlElementPtr ), attribute->name ) );
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
                    AddNodes (
                        aNode->AddNode (
                            Construct (
                                reinterpret_cast<const char*> ( element->name ),
                                ExtractElementAttributes ( element ),
                                aNode )
                        ), node->children );
                }
                else if ( xmlNodeIsText ( node ) && !xmlIsBlankNode ( node ) )
                {
                    AddNodes ( aNode->AddNode ( new Text { reinterpret_cast<const char*> ( node->content ), aNode } ), node->children );
                }
            }
        }

        Document::Document () = default;

        css_error resolve_url ( void *pw,
                                const char *base, lwc_string *rel, lwc_string **abs )
        {
            ( void ) pw;
            ( void ) base;

            /* About as useless as possible */
            *abs = lwc_string_ref ( rel );
            return CSS_OK;
        }

        Document::Document ( const std::string& aFilename )
        {
            xmlDocPtr document{xmlReadFile ( aFilename.c_str(), nullptr, 0 ) };
            if ( document == nullptr )
            {
                throw std::runtime_error ( "Could not parse xml file" );
            }

            css_error code{};
            css_stylesheet_params params{};
            params.params_version = CSS_STYLESHEET_PARAMS_VERSION_1;
            params.level = CSS_LEVEL_21;
            params.charset = "UTF-8";
            params.url = aFilename.c_str();
            params.title = aFilename.c_str();
            params.allow_quirks = false;
            params.inline_style = false;
            params.resolve = resolve_url;
            params.resolve_pw = nullptr;
            params.import = nullptr;
            params.import_pw = nullptr;
            params.color = nullptr;
            params.color_pw = nullptr;
            params.font = nullptr;
            params.font_pw = nullptr;

            {
                css_stylesheet* stylesheet{};
                code = css_stylesheet_create ( &params, &stylesheet );
                if ( code != CSS_OK )
                {
                    throw std::runtime_error ( css_error_to_string ( code ) );
                }
                mStyleSheet.reset ( stylesheet );
            }

            ///@todo use document->children instead?
            xmlElementPtr root_element = reinterpret_cast<xmlElementPtr> ( xmlDocGetRootElement ( document ) );
            mDocumentElement = Construct ( reinterpret_cast<const char*> ( root_element->name ), ExtractElementAttributes ( root_element ), nullptr );
            AddNodes ( mDocumentElement, root_element->children );
            xmlFreeDoc ( document );
            /**@todo Emit onload event.*/
        }

        void Document::Load()
        {
            mDocumentElement->TraverseDepthFirstPreOrder (
                [] ( Node * aNode )
            {
                aNode->Load();
            } );
        }

        void Document::Unload ()
        {
            mDocumentElement->TraverseDepthFirstPreOrder (
                [] ( Node * aNode )
            {
                aNode->Unload ();
            } );
        }

        Document::~Document() = default;

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
}