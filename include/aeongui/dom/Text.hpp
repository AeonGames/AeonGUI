/*
Copyright (C) 2020,2024,2025 Rodrigo Jose Hernandez Cordoba

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
        class Text : public Node
        {
        public:
            DLL Text ( const std::string& aText, Node* aParent );
            DLL ~Text() final;
            /**DOM Properties and Methods @{*/
            NodeType nodeType() const final;
            std::string wholeText() const;
            /**@}*/
        private:
            std::string mText{};
        };
    }
}
#endif
