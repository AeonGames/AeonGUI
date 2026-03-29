/*
Copyright (C) 2019,2020,2023-2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_NODE_H
#define AEONGUI_NODE_H
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <variant>
#include "aeongui/Platform.hpp"
#include "aeongui/AttributeMap.hpp"
#include "aeongui/dom/EventTarget.hpp"

namespace AeonGUI
{
    class Canvas;
    class Document;
    namespace DOM
    {
        /** @brief Base class for all nodes in the DOM tree.
         *
         *  Implements the DOM Node interface: parent/child relationships,
         *  tree traversal, and draw hooks for rendering.
         *  @see https://dom.spec.whatwg.org/#interface-node
         */
        class Node : public EventTarget
        {
        public:
            /** @brief DOM node type constants. */
            enum NodeType
            {
                ELEMENT_NODE = 1,               ///< An Element node.
                ATTRIBUTE_NODE = 2,             ///< An Attribute node (legacy).
                TEXT_NODE = 3,                  ///< A Text node.
                CDATA_SECTION_NODE = 4,         ///< A CDATASection node.
                ENTITY_REFERENCE_NODE = 5,      ///< An EntityReference node (legacy).
                ENTITY_NODE = 6,                ///< An Entity node (legacy).
                PROCESSING_INSTRUCTION_NODE = 7,///< A ProcessingInstruction node.
                COMMENT_NODE = 8,               ///< A Comment node.
                DOCUMENT_NODE = 9,              ///< A Document node.
                DOCUMENT_TYPE_NODE = 10,        ///< A DocumentType node.
                DOCUMENT_FRAGMENT_NODE = 11,    ///< A DocumentFragment node.
                NOTATION_NODE = 12,             ///< A Notation node (legacy).
            };
            /** @brief Construct a node with an optional parent.
             *  @param aParent The parent node, or nullptr.
             */
            DLL Node ( Node* aParent = nullptr );
            /** @brief Add a child node.
             *  @param aNode The child to add (ownership transferred).
             *  @return Raw pointer to the added node.
             */
            DLL Node* AddNode ( std::unique_ptr<Node> aNode );
            /** @brief Remove a child node.
             *  @param aNode Raw pointer to the child to remove.
             *  @return Ownership of the removed node.
             */
            DLL std::unique_ptr<Node> RemoveNode ( const Node* aNode );
            /** @brief Traverse the tree depth-first in pre-order.
             *  @param aAction Action invoked for each node.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aAction );
            /** @brief Traverse the tree depth-first in pre-order (const).
             *  @param aAction Action invoked for each node.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aAction ) const;
            /** @brief Traverse the tree depth-first in post-order.
             *  @param aAction Action invoked for each node.
             */
            DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Node& ) >& aAction );
            /** @brief Traverse the tree depth-first in post-order (const).
             *  @param aAction Action invoked for each node.
             */
            DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Node& ) >& aAction ) const;
            /** @brief Traverse pre-order with separate pre and post callbacks.
             *  @param aPreamble  Called before visiting children.
             *  @param aPostamble Called after visiting children.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble );
            /** @brief Traverse pre-order with pre/post callbacks (const).
             *  @param aPreamble  Called before visiting children.
             *  @param aPostamble Called after visiting children.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble ) const;
            /** @brief Traverse pre-order with pre/post callbacks and a predicate filter.
             *  @param aPreamble       Called before visiting children.
             *  @param aPostamble      Called after visiting children.
             *  @param aUnaryPredicate Only descend into children for which this returns true.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble, const std::function<bool ( Node& ) >& aUnaryPredicate );
            /** @brief Traverse pre-order with pre/post callbacks and predicate (const).
             *  @param aPreamble       Called before visiting children.
             *  @param aPostamble      Called after visiting children.
             *  @param aUnaryPredicate Only descend into children for which this returns true.
             */
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble, const std::function<bool ( const Node& ) >& aUnaryPredicate ) const;

            /** @brief Stack-based depth-first pre-order traversal (reentrant-safe).
             *  @param aAction Action invoked for each node.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aAction );
            /** @brief Stack-based depth-first pre-order traversal (const, reentrant-safe).
             *  @param aAction Action invoked for each node.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aAction ) const;
            /** @brief Stack-based depth-first post-order traversal (reentrant-safe).
             *  @param aAction Action invoked for each node.
             */
            DLL void StackTraverseDepthFirstPostOrder ( const std::function<void ( Node& ) >& aAction );
            /** @brief Stack-based depth-first post-order traversal (const, reentrant-safe).
             *  @param aAction Action invoked for each node.
             */
            DLL void StackTraverseDepthFirstPostOrder ( const std::function<void ( const Node& ) >& aAction ) const;
            /** @brief Stack-based pre-order with pre/post callbacks (reentrant-safe).
             *  @param aPreamble  Called before visiting children.
             *  @param aPostamble Called after visiting children.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble );
            /** @brief Stack-based pre-order with pre/post callbacks (const, reentrant-safe).
             *  @param aPreamble  Called before visiting children.
             *  @param aPostamble Called after visiting children.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble ) const;
            /** @brief Stack-based pre-order with pre/post callbacks and predicate (reentrant-safe).
             *  @param aPreamble       Called before visiting children.
             *  @param aPostamble      Called after visiting children.
             *  @param aUnaryPredicate Only descend into children for which this returns true.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( Node& ) >& aPreamble, const std::function<void ( Node& ) >& aPostamble, const std::function<bool ( Node& ) >& aUnaryPredicate );
            /** @brief Stack-based pre-order with pre/post/predicate (const, reentrant-safe).
             *  @param aPreamble       Called before visiting children.
             *  @param aPostamble      Called after visiting children.
             *  @param aUnaryPredicate Only descend into children for which this returns true.
             */
            DLL void StackTraverseDepthFirstPreOrder ( const std::function<void ( const Node& ) >& aPreamble, const std::function<void ( const Node& ) >& aPostamble, const std::function<bool ( const Node& ) >& aUnaryPredicate ) const;

            /** @brief Update animation state for this node.
             *  @param aDocumentTime The current document time in seconds.
             */
            DLL virtual void Update ( double aDocumentTime );
            /** @brief Begin drawing this node on the canvas.
             *  @param aCanvas The target canvas.
             */
            DLL virtual void DrawStart ( Canvas& aCanvas ) const;
            /** @brief Finish drawing this node on the canvas.
             *  @param aCanvas The target canvas.
             */
            DLL virtual void DrawFinish ( Canvas& aCanvas ) const;
            /**
             * Use OnLoad to implement custom loading behavior in derived classes.
             * This method is called after all nodes are created and all children
             * have been set.
             * Nodes are visited in depth-first pre-order.
             */
            DLL virtual void OnLoad ();
            /**
             * Use OnUnload to implement custom unloading behavior in derived classes.
             * This method is called while all nodes are still in place but before
             * they are destroyed.
             * Nodes are visited in depth-first post-order.
             */
            DLL virtual void OnUnload ();
            /** Returns whether this node and all descendants should be skipped
             *  in a drawing operation.
             *  @return true by default override to disable drawing.
            */
            DLL virtual bool IsDrawEnabled() const;
            DLL virtual ~Node();
            /**DOM Properties and Methods @{*/
            /** @brief Get the parent node.
             *  @return Pointer to the parent, or nullptr if this is the root.
             */
            DLL Node* parentNode() const;
            /** @brief Get the parent element (same as parentNode for elements).
             *  @return Pointer to the parent element, or nullptr.
             */
            DLL Node* parentElement() const;
            /** @brief Get the node type.
             *  @return One of the NodeType constants.
             */
            virtual NodeType nodeType() const = 0;
            /** @brief Get the list of child nodes.
             *  @return Const reference to the vector of children.
             */
            const std::vector<std::unique_ptr<Node>>& childNodes() const;
            /**@}*/
        private:
            DLL virtual void OnAncestorChanged();
            Node* mParent{};
            std::vector<std::unique_ptr<Node>> mChildren{};
            mutable std::vector<std::unique_ptr<Node>>::size_type mIterator{ 0 };
        };
    }
}
#endif
