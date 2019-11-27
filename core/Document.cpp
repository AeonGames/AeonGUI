/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
    static void AddElements ( Document* aDocument, xmlNode * aNode )
    {
        for ( xmlNode * node = aNode; node; node = node->next )
        {
            if ( node->type == XML_ELEMENT_NODE )
            {
                aDocument->AddElement ( Construct ( reinterpret_cast<xmlElementPtr> ( node ) ) );
            }
            AddElements ( aDocument, node->children );
        }
    }

    Document::Document () = default;

    Document::Document ( const std::string& aFilename ) :
        mDocument{xmlReadFile ( aFilename.c_str(), nullptr, 0 ) }
    {
        if ( mDocument == nullptr )
        {
            throw std::runtime_error ( "Could not parse xml file" );
        }
        AddElements ( this, xmlDocGetRootElement ( mDocument ) );
    }

    Document::~Document()
    {
        if ( mDocument != nullptr )
        {
            xmlFreeDoc ( mDocument );
            mDocument = nullptr;
        }
    }

    Element* Document::AddElement ( std::unique_ptr<Element> aElement )
    {
        return mChildren.emplace_back ( std::move ( aElement ) ).get();
    }

    std::unique_ptr<Element> Document::RemoveElement ( const Element* aElement )
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
        return result;
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aAction )
    {
        for ( auto & mRootElement : mChildren )
        {
            mRootElement->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aAction ) const
    {
        for ( const auto& mRootElement : mChildren )
        {
            static_cast<const Element*> ( mRootElement.get() )->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPostOrder ( const std::function<void ( Element& ) >& aAction )
    {
        for ( auto & mRootElement : mChildren )
        {
            mRootElement->TraverseDepthFirstPostOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPostOrder ( const std::function<void ( const Element& ) >& aAction ) const
    {
        for ( const auto& mRootElement : mChildren )
        {
            static_cast<const Element*> ( mRootElement.get() )->TraverseDepthFirstPostOrder ( aAction );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aPreamble, const std::function<void ( Element& ) >& aPostamble )
    {
        for ( auto & mRootElement : mChildren )
        {
            mRootElement->TraverseDepthFirstPreOrder ( aPreamble, aPostamble );
        }
    }

    void Document::TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aPreamble, const std::function<void ( const Element& ) >& aPostamble ) const
    {
        for ( const auto& mRootElement : mChildren )
        {
            static_cast<const Element*> ( mRootElement.get() )->TraverseDepthFirstPreOrder ( aPreamble, aPostamble );
        }
    }

    void Document::Draw ( Canvas& aCanvas ) const
    {
        TraverseDepthFirstPreOrder (
            [&aCanvas] ( const Element & aElement )
        {
            aElement.DrawStart ( aCanvas );
        },
        [&aCanvas] ( const Element & aElement )
        {
            aElement.DrawFinish ( aCanvas );
        } );
    }
    void Document::Load ( JavaScript& aJavaScript )
    {
        TraverseDepthFirstPreOrder (
            [&aJavaScript] ( Element & aElement )
        {
            aElement.Load ( aJavaScript );
        } );
    }
    void Document::Unload ( JavaScript& aJavaScript )
    {
        TraverseDepthFirstPreOrder (
            [&aJavaScript] ( Element & aElement )
        {
            aElement.Unload ( aJavaScript );
        } );
    }
}
