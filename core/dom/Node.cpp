/******************************************************************************
Copyright (C) 2010-2013,2019,2020,2023-2026 Rodrigo Hernandez Cordoba

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
#include <stack>
#include "aeongui/dom/Node.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/Text.hpp"
#include "aeongui/CSSSelector.hpp"
#include "aeongui/Color.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        Node::Node ( Node* aParent ) : mParent{aParent} {};
        Node::~Node() = default;

        const std::vector<std::unique_ptr<Node>>& Node::childNodes() const
        {
            return mChildren;
        }

        Document* Node::ownerDocument() const
        {
            Node* node = mParent;
            while ( node )
            {
                if ( node->nodeType() == NodeType::DOCUMENT_NODE )
                {
                    return static_cast<Document*> ( node );
                }
                node = node->parentNode();
            }
            return nullptr;
        }

        Element* Node::querySelector ( const DOMString& aSelector ) const
        {
            auto selectors = ParseSelector ( aSelector );
            Element* result = nullptr;
            const Node* root = this;
            StackTraverseDepthFirstPreOrder (
                [&selectors, &result, root] ( const Node & aNode )
            {
                if ( !result && &aNode != root && aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    Element* elem = const_cast<Element*> ( static_cast<const Element*> ( &aNode ) );
                    if ( MatchesAny ( *elem, selectors ) )
                    {
                        result = elem;
                    }
                }
            } );
            return result;
        }

        std::vector<Element*> Node::querySelectorAll ( const DOMString& aSelector ) const
        {
            auto selectors = ParseSelector ( aSelector );
            std::vector<Element*> results;
            const Node* root = this;
            StackTraverseDepthFirstPreOrder (
                [&selectors, &results, root] ( const Node & aNode )
            {
                if ( &aNode != root && aNode.nodeType() == Node::ELEMENT_NODE )
                {
                    Element* elem = const_cast<Element*> ( static_cast<const Element*> ( &aNode ) );
                    if ( MatchesAny ( *elem, selectors ) )
                    {
                        results.push_back ( elem );
                    }
                }
            } );
            return results;
        }

        bool Node::IsDrawEnabled() const
        {
            return true;
        }

        void Node::Update ( double aDocumentTime )
        {
            ( void ) aDocumentTime;
        }

        void Node::DrawStart ( Canvas& aCanvas ) const
        {
            // Do nothing by default
            ( void ) aCanvas;
        }

        void Node::DrawFinish ( Canvas& aCanvas ) const
        {
            // Do nothing by default
            ( void ) aCanvas;
        }

        void Node::OnLoad ()
        {
            // Do nothing by default
        }

        void Node::OnUnload ()
        {
            // Do nothing by default
        }

        Node* Node::parentNode() const
        {
            return mParent;
        }
        Node* Node::parentElement() const
        {
            if ( mParent && mParent->nodeType() == ELEMENT_NODE )
            {
                return mParent;
            }
            return nullptr;
        }

        /*  This is ugly, but it is only way to use the same code for the const and the non const version
            without having to add template or friend members to the class declaration. */
#define TraverseDepthFirstPreOrder(...) \
    void Node::TraverseDepthFirstPreOrder ( const std::function<void ( __VA_ARGS__ Node& ) >& aAction ) __VA_ARGS__ \
    {\
        auto node{this};\
        aAction ( *node );\
        auto parent = mParent;\
        while ( node != parent )\
        {\
            if ( node->mIterator < node->mChildren.size() )\
            {\
                auto prev = node;\
                node = node->mChildren[node->mIterator].get();\
                aAction ( *node );\
                prev->mIterator++;\
            }\
            else\
            {\
                node->mIterator = 0; /* Reset counter for next traversal.*/\
                node = node->mParent;\
            }\
        }\
    }

        TraverseDepthFirstPreOrder ( const )
        TraverseDepthFirstPreOrder( )
#undef TraverseDepthFirstPreOrder

#define TraverseDepthFirstPostOrder(...) \
    void Node::TraverseDepthFirstPostOrder ( const std::function<void ( __VA_ARGS__ Node& ) >& aAction ) __VA_ARGS__ \
    { \
        /* \
        This code implements a similar solution to this stackoverflow answer: \
        http://stackoverflow.com/questions/5987867/traversing-a-n-ary-tree-without-using-recurrsion/5988138#5988138 \
        */ \
        auto node{this}; \
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
    void Node::TraverseDepthFirstPreOrder ( \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPreamble, \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPostamble ) __VA_ARGS__ \
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

#define TraverseDepthFirstPostOrder(...) \
    void Node::TraverseDepthFirstPreOrder ( \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPreamble, \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPostamble, \
        const std::function<bool ( __VA_ARGS__ Node& ) >& aUnaryPredicate ) __VA_ARGS__ \
    { \
        if(!aUnaryPredicate(*this)){return;} \
        auto node = this; \
        aPreamble ( *node ); \
        auto parent = mParent; \
        while ( node != parent ) \
        { \
            if ( node->mIterator < node->mChildren.size() && aUnaryPredicate(*node)) \
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

        /* Stack-based traversal implementations - reentrant-safe alternatives
           that use an explicit std::stack instead of the per-node mIterator. */

#define StackTraverseDepthFirstPreOrder(...) \
    void Node::StackTraverseDepthFirstPreOrder ( const std::function<void ( __VA_ARGS__ Node& ) >& aAction ) __VA_ARGS__ \
    { \
        struct Frame { __VA_ARGS__ Node* node; size_t childIndex; }; \
        std::stack<Frame> stack; \
        stack.push ( Frame{this, 0} ); \
        aAction ( *this ); \
        while ( !stack.empty() ) \
        { \
            auto& frame = stack.top(); \
            if ( frame.childIndex < frame.node->mChildren.size() ) \
            { \
                __VA_ARGS__ Node* child = frame.node->mChildren[frame.childIndex].get(); \
                ++frame.childIndex; \
                stack.push ( Frame{child, 0} ); \
                aAction ( *child ); \
            } \
            else \
            { \
                stack.pop(); \
            } \
        } \
    }

        StackTraverseDepthFirstPreOrder ( const )
        StackTraverseDepthFirstPreOrder( )
#undef StackTraverseDepthFirstPreOrder

#define StackTraverseDepthFirstPostOrder(...) \
    void Node::StackTraverseDepthFirstPostOrder ( const std::function<void ( __VA_ARGS__ Node& ) >& aAction ) __VA_ARGS__ \
    { \
        struct Frame { __VA_ARGS__ Node* node; size_t childIndex; }; \
        std::stack<Frame> stack; \
        stack.push ( Frame{this, 0} ); \
        while ( !stack.empty() ) \
        { \
            auto& frame = stack.top(); \
            if ( frame.childIndex < frame.node->mChildren.size() ) \
            { \
                __VA_ARGS__ Node* child = frame.node->mChildren[frame.childIndex].get(); \
                ++frame.childIndex; \
                stack.push ( Frame{child, 0} ); \
            } \
            else \
            { \
                aAction ( *frame.node ); \
                stack.pop(); \
            } \
        } \
    }

        StackTraverseDepthFirstPostOrder ( const )
        StackTraverseDepthFirstPostOrder( )
#undef StackTraverseDepthFirstPostOrder

#define StackTraverseDepthFirstPostOrder(...) \
    void Node::StackTraverseDepthFirstPreOrder ( \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPreamble, \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPostamble ) __VA_ARGS__ \
    { \
        struct Frame { __VA_ARGS__ Node* node; size_t childIndex; }; \
        std::stack<Frame> stack; \
        stack.push ( Frame{this, 0} ); \
        aPreamble ( *this ); \
        while ( !stack.empty() ) \
        { \
            auto& frame = stack.top(); \
            if ( frame.childIndex < frame.node->mChildren.size() ) \
            { \
                __VA_ARGS__ Node* child = frame.node->mChildren[frame.childIndex].get(); \
                ++frame.childIndex; \
                stack.push ( Frame{child, 0} ); \
                aPreamble ( *child ); \
            } \
            else \
            { \
                aPostamble ( *frame.node ); \
                stack.pop(); \
            } \
        } \
    }

        StackTraverseDepthFirstPostOrder ( const )
        StackTraverseDepthFirstPostOrder( )
#undef StackTraverseDepthFirstPostOrder

#define StackTraverseDepthFirstPostOrder(...) \
    void Node::StackTraverseDepthFirstPreOrder ( \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPreamble, \
        const std::function<void ( __VA_ARGS__ Node& ) >& aPostamble, \
        const std::function<bool ( __VA_ARGS__ Node& ) >& aUnaryPredicate ) __VA_ARGS__ \
    { \
        if ( !aUnaryPredicate ( *this ) ) { return; } \
        struct Frame { __VA_ARGS__ Node* node; size_t childIndex; }; \
        std::stack<Frame> stack; \
        stack.push ( Frame{this, 0} ); \
        aPreamble ( *this ); \
        while ( !stack.empty() ) \
        { \
            auto& frame = stack.top(); \
            if ( frame.childIndex < frame.node->mChildren.size() && aUnaryPredicate ( *frame.node ) ) \
            { \
                __VA_ARGS__ Node* child = frame.node->mChildren[frame.childIndex].get(); \
                ++frame.childIndex; \
                stack.push ( Frame{child, 0} ); \
                aPreamble ( *child ); \
            } \
            else \
            { \
                aPostamble ( *frame.node ); \
                stack.pop(); \
            } \
        } \
    }

        StackTraverseDepthFirstPostOrder ( const )
        StackTraverseDepthFirstPostOrder( )
#undef StackTraverseDepthFirstPostOrder

        DOMString Node::textContent() const
        {
            switch ( nodeType() )
            {
            case DOCUMENT_NODE:
            case DOCUMENT_TYPE_NODE:
                return {};
            case TEXT_NODE:
                return static_cast<const Text*> ( this )->data();
            default:
            {
                DOMString result;
                for ( const auto& child : mChildren )
                {
                    if ( child->nodeType() == TEXT_NODE )
                    {
                        result += static_cast<const Text*> ( child.get() )->data();
                    }
                    else if ( child->nodeType() != DOCUMENT_TYPE_NODE )
                    {
                        result += child->textContent();
                    }
                }
                return result;
            }
            }
        }

        void Node::setTextContent ( const DOMString& aTextContent )
        {
            switch ( nodeType() )
            {
            case DOCUMENT_NODE:
            case DOCUMENT_TYPE_NODE:
                return;
            default:
            {
                // Find the first Text child and remove any extras.
                auto textNodeIt = mChildren.begin();
                for ( ; textNodeIt != mChildren.end(); ++textNodeIt )
                {
                    if ( ( *textNodeIt )->nodeType() == TEXT_NODE )
                    {
                        mChildren.erase ( std::remove_if ( textNodeIt + 1, mChildren.end(),
                                                           [] ( const std::unique_ptr<Node>& node )
                        {
                            return node->nodeType() == TEXT_NODE;
                        } ), mChildren.end() );
                        break;
                    }
                }
                // Only add a new Text node if the given content is non-empty and there isn't already one.
                if ( textNodeIt != mChildren.end() )
                {
                    static_cast<Text*> ( textNodeIt->get() )->setData ( aTextContent );
                }
                else if ( !aTextContent.empty() )
                {
                    AddNode ( std::make_unique<Text> ( aTextContent, this ) );
                }
                Document* doc = ownerDocument();
                if ( doc )
                {
                    doc->MarkDirty();
                }
                break;
            }
            }
        }

        void Node::OnAncestorChanged()
        {
            // Do nothing by default
        }

        void Node::OnInsertedIntoDocument ( Document& )
        {
            // Do nothing by default — Element overrides to maintain the
            // document's id index.
        }

        void Node::OnRemovedFromDocument ( Document& )
        {
            // Do nothing by default — Element overrides to maintain the
            // document's id index.
        }

        void Node::OnChildInserted ( Node& )
        {
            // Do nothing by default — Element overrides to maintain its
            // animation-children cache.
        }

        void Node::OnChildRemoved ( Node& )
        {
            // Do nothing by default — Element overrides to maintain its
            // animation-children cache.
        }

        Node* Node::AddNode ( std::unique_ptr<Node> aNode )
        {
            aNode->mParent = this;
            Node* node { mChildren.emplace_back ( std::move ( aNode ) ).get() };
            OnChildInserted ( *node );
            if ( Document * doc = node->ownerDocument() )
            {
                node->TraverseDepthFirstPreOrder ( [doc] ( Node & n )
                {
                    n.OnInsertedIntoDocument ( *doc );
                } );
            }
            node->TraverseDepthFirstPreOrder (  (
                                                    [] ( Node & node )
            {
                node.OnAncestorChanged();
            } ) );
            return node;
        }

        std::unique_ptr<Node> Node::RemoveNode ( const Node* aNode )
        {
            std::unique_ptr<Node> node{};
            auto i = std::find_if ( mChildren.begin(), mChildren.end(), [aNode] ( const std::unique_ptr<Node>& node )
            {
                return aNode == node.get();
            } );
            if ( i != mChildren.end() )
            {
                // Notify the parent before detaching so any cached
                // pointers to the child can be cleared while the child is
                // still owned by the child list.
                OnChildRemoved ( **i );
                node = std::move ( *i );
                mChildren.erase ( i );
            }
            // While the subtree's mParent still points at us we can locate
            // its owning Document; do the unregister pass first so id
            // mappings are torn down before the subtree is detached.
            if ( node )
            {
                if ( Document * doc = node->ownerDocument() )
                {
                    node->TraverseDepthFirstPreOrder ( [doc] ( Node & n )
                    {
                        n.OnRemovedFromDocument ( *doc );
                    } );
                }
                node->mParent = nullptr;
                node->TraverseDepthFirstPreOrder (  (
                                                        [] ( Node & node )
                {
                    node.OnAncestorChanged();
                } ) );
            }
            return node;
        }
    }
}