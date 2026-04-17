/*
Copyright (C) 2020,2024,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_TEXT_H
#define AEONGUI_TEXT_H
#include <string>
#include "aeongui/Platform.hpp"
#include "Node.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Represents a DOM Text node.
         *
         *  Contains character data as a child of an Element.
         */
        class Text : public Node
        {
        public:
            /** @brief Construct a Text node.
             *  @param aText   The text content.
             *  @param aParent The parent node.
             */
            AEONGUI_DLL Text ( const std::string& aText, Node* aParent );
            /** @brief Destructor. */
            AEONGUI_DLL ~Text() final;
            /**DOM Properties and Methods @{*/
            /** @brief Get the node type (always TEXT_NODE).
             *  @return NodeType::TEXT_NODE. */
            NodeType nodeType() const final;
            /** @brief Get the concatenated text of this and adjacent text nodes.
             *  @return The whole text content.
             */
            std::string wholeText() const;
            /** @brief Get the character data of this text node.
             *  @return The text data.
             */
            AEONGUI_DLL const std::string& data() const;
            /** @brief Set the character data of this text node.
             *  @param aData The new text data.
             */
            AEONGUI_DLL void setData ( const std::string& aData );
            /**@}*/
        private:
            std::string mText{};
        };
    }
}
#endif
