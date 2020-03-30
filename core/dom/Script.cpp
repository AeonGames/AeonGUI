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
#include <functional>
#include <iostream>
#include "Script.h"
#include "Text.h"
#include "aeongui/JavaScript.h"

namespace AeonGUI
{
    namespace DOM
    {
        Script::Script ( const std::string& aTagName, const AttributeMap& aAttributes ) : Element ( aTagName, aAttributes )
        {
        }
        Script::~Script()
        {
        }
        void Script::Load ()
        {
#if 0
            const auto& children = childNodes();
            ///@todo Don't asume script elements don't contain more elements or more than one text node.
            auto text_node = std::find_if ( children.begin(), children.end(), [] ( const std::unique_ptr<Node>& aNode )
            {
                return aNode->nodeType() == TEXT_NODE;
            } );
            if ( text_node != children.end() )
            {
                aJavaScript.Eval ( reinterpret_cast<const Text*> ( text_node->get() )->wholeText() );
            }
#endif
        }
        void Script::Unload ()
        {
        }
    }
}
