/*
Copyright (C) 2019,2020,2023-2025 Rodrigo Jose Hernandez Cordoba

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

namespace AeonGUI
{
    class Canvas;
    class Document;
    namespace DOM
    {
        class Node
        {
        public:
            enum NodeType
            {
                ELEMENT_NODE = 1,
                ATTRIBUTE_NODE = 2,
                TEXT_NODE = 3,
                CDATA_SECTION_NODE = 4,
                ENTITY_REFERENCE_NODE = 5,
                ENTITY_NODE = 6,
                PROCESSING_INSTRUCTION_NODE = 7,
                COMMENT_NODE = 8,
                DOCUMENT_NODE = 9,
                DOCUMENT_TYPE_NODE = 10,
                DOCUMENT_FRAGMENT_NODE = 11,
                NOTATION_NODE = 12,
            };
            DLL Node ( Node* aParent = nullptr );
            DLL Node* AddNode ( Node* aNode );
            DLL Node* RemoveNode ( const Node* aNode );
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node* ) >& aAction );
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node* ) >& aAction ) const;
            DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Node* ) >& aAction );
            DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Node* ) >& aAction ) const;
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node* ) >& aPreamble, const std::function<void ( Node* ) >& aPostamble );
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node* ) >& aPreamble, const std::function<void ( const Node* ) >& aPostamble ) const;
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Node* ) >& aPreamble, const std::function<void ( Node* ) >& aPostamble, const std::function<bool ( Node* ) >& aUnaryPredicate );
            DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Node* ) >& aPreamble, const std::function<void ( const Node* ) >& aPostamble, const std::function<bool ( const Node* ) >& aUnaryPredicate ) const;

            DLL virtual void DrawStart ( Canvas& aCanvas ) const;
            DLL virtual void DrawFinish ( Canvas& aCanvas ) const;
            DLL virtual void Load ();
            DLL virtual void Unload ();
            /** Returns whether this node and all descendants should be skipped
             *  in a drawing operation.
             *  @return true by default override to disable drawing.
            */
            DLL virtual bool IsDrawEnabled() const;
            DLL virtual ~Node();
            /**DOM Properties and Methods @{*/
            DLL Node* parentNode() const;
            DLL Node* parentElement() const;
            virtual NodeType nodeType() const = 0;
            const std::vector<Node*>& childNodes() const;
            /**@}*/
        private:
            DLL virtual void OnAncestorChanged();
            Node* mParent{};
            std::vector<Node*> mChildren{};
            mutable std::vector<Node*>::size_type mIterator{ 0 };
        };
    }
}
#endif
