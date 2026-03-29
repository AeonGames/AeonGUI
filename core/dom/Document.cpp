/*
Copyright (C) 2019,2020,2023,2024-2026 Rodrigo Jose Hernandez Cordoba

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
#include <stack>
#include "aeongui/ElementFactory.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/SVGGeometryElement.hpp"
#include "aeongui/dom/Text.hpp"

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
                    AddNodes ( aNode->AddNode ( std::make_unique<Text> ( reinterpret_cast<const char*> ( node->content ), aNode ) ), node->children );
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

        Node::NodeType Document::nodeType() const
        {
            return NodeType::DOCUMENT_NODE;
        }

        void Document::Load ( const USVString& aFilename )
        {
            mUrl = aFilename;
            xmlDocPtr document{xmlReadFile ( reinterpret_cast<const char*> ( aFilename.c_str() ), nullptr, 0 ) };
            if ( document == nullptr )
            {
                throw std::runtime_error ( "Could not open file: " + aFilename );
            }

            css_error code{};
            css_stylesheet_params params{};
            params.params_version = CSS_STYLESHEET_PARAMS_VERSION_1;
            params.level = CSS_LEVEL_3;
            params.charset = "UTF-8";
            params.url = reinterpret_cast<const char*> ( aFilename.c_str() );
            params.title = reinterpret_cast<const char*> ( aFilename.c_str() );
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
            AddNodes ( AddNode ( Construct ( reinterpret_cast<const char*> ( root_element->name ), ExtractElementAttributes ( root_element ), this ) ), root_element->children );
            xmlFreeDoc ( document );

            // Parse <style> element content into the document stylesheet
            TraverseDepthFirstPreOrder (
                [this] ( Node & aNode )
            {
                if ( aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    Element* elem = static_cast<Element*> ( &aNode );
                    if ( elem->tagName() == "style" )
                    {
                        for ( auto& child : aNode.childNodes() )
                        {
                            if ( child->nodeType() == Node::TEXT_NODE )
                            {
                                Text* text = static_cast<Text*> ( child.get() );
                                std::string cssText = text->wholeText();
                                css_stylesheet_append_data ( mStyleSheet.get(),
                                                             reinterpret_cast<const uint8_t*> ( cssText.data() ),
                                                             cssText.size() );
                            }
                        }
                    }
                }
            } );
            css_stylesheet_data_done ( mStyleSheet.get() );

            // Re-select CSS for all elements using the document stylesheet
            TraverseDepthFirstPreOrder (
                [this] ( Node & aNode )
            {
                if ( aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    static_cast<Element*> ( &aNode )->ReselectCSS ( mStyleSheet.get() );
                }
            } );

            Load();
        }

        const USVString& Document::url() const
        {
            return mUrl;
        }

        css_stylesheet* Document::GetStyleSheet() const
        {
            return mStyleSheet.get();
        }

        void Document::Load()
        {
            TraverseDepthFirstPreOrder (
                [] ( Node & aNode )
            {
                aNode.OnLoad();
            } );
        }

        void Document::Unload ()
        {
            TraverseDepthFirstPostOrder (
                [] ( Node & aNode )
            {
                aNode.OnUnload ();
            } );
        }

        Document::~Document()
        {
            Unload();
        }

        void Document::AdvanceTime ( double aDeltaTime )
        {
            mDocumentTime += aDeltaTime;
            TraverseDepthFirstPreOrder (
                [docTime = mDocumentTime] ( Node & aNode )
            {
                aNode.Update ( docTime );
            } );
        }

        void Document::Draw ( Canvas& aCanvas ) const
        {
            TraverseDepthFirstPreOrder (
                [&aCanvas] ( const Node & aNode )
            {
                aCanvas.Save();
                aNode.DrawStart ( aCanvas );
            },
            [&aCanvas] ( const Node & aNode )
            {
                aNode.DrawFinish ( aCanvas );
                aCanvas.Restore();
            },
            [] ( const Node & aNode )
            {
                return aNode.IsDrawEnabled();
            } );
        }

        Element* Document::getElementById ( const DOMString& aElementId ) const
        {
            Element* result = nullptr;
            StackTraverseDepthFirstPreOrder (
                [&aElementId, &result] ( const Node & aNode )
            {
                if ( !result && aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    Element* elem = const_cast<Element*> ( static_cast<const Element*> ( &aNode ) );
                    if ( elem->id() == aElementId )
                    {
                        result = elem;
                    }
                }
            } );
            return result;
        }

        Element* Document::elementFromPoint ( Canvas& aCanvas, double aX, double aY ) const
        {
            Element* result = nullptr;
            TraverseDepthFirstPreOrder (
                [&aCanvas, aX, aY, &result] ( const Node & aNode )
            {
                aCanvas.Save();
                aNode.DrawStart ( aCanvas );
                if ( aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    const auto* geometry = dynamic_cast<const SVGGeometryElement*> ( &aNode );
                    if ( geometry && aCanvas.PointInPath ( geometry->GetPath(), aX, aY ) )
                    {
                        result = const_cast<Element*> ( static_cast<const Element*> ( &aNode ) );
                    }
                }
            },
            [&aCanvas] ( const Node & aNode )
            {
                aCanvas.Restore();
            },
            [] ( const Node & aNode )
            {
                return aNode.IsDrawEnabled();
            } );
            return result;
        }
    }
}