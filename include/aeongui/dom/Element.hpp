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
#ifndef AEONGUI_ELEMENT_H
#define AEONGUI_ELEMENT_H
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <variant>
#include "aeongui/Platform.hpp"
#include "aeongui/AttributeMap.hpp"
#include "aeongui/StyleSheet.hpp"
#include "aeongui/dom/DOMString.hpp"
#include "Node.hpp"

extern "C"
{
    typedef struct lwc_string_s lwc_string;
}
namespace AeonGUI
{
    class Canvas;
    namespace DOM
    {
        class Document;
        class Element : public Node
        {
        public:
            DLL Element ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent );
            DLL virtual ~Element();
            /**DOM Properties and Methods @{*/
            NodeType nodeType() const final;
            const DOMString& tagName() const;
            const DOMString& id() const;
            const std::vector<lwc_string*>& classes() const;
            /**@}*/
        private:
            DOMString mTagName{};
            DOMString mId{};
            std::vector<lwc_string*> mClasses{};
            DLL void OnAncestorChanged() override;
        protected:
            StyleSheetPtr mInlineStyleSheet{};
            SelectResultsPtr mComputedStyles{};
            css_select_results* GetParentComputedStyles() const;
            css_select_results* GetComputedStyles() const;
        };
    }
}
#endif
