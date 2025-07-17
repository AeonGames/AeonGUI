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
#include "aeongui/Platform.h"
#include "aeongui/AttributeMap.h"
#include "aeongui/StyleSheet.h"
#include "aeongui/dom/DOMString.hpp"
#include "Node.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Canvas;
        class JavaScript;
        class Document;
        class Element : public Node
        {
        public:
            DLL Element ( const DOMString& aTagName, const AttributeMap& aAttributes, Node* aParent );
            //DLL AttributeType GetAttribute ( const char* attrName, const AttributeType& aDefault = {} ) const;
            //DLL AttributeType GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault = {} ) const;
            //DLL void SetAttribute ( const char* attrName, const AttributeType& aValue );
            DLL virtual ~Element();
            /**DOM Properties and Methods @{*/
            NodeType nodeType() const final;
            const DOMString& tagName() const;
            const DOMString& id() const;
            const std::vector<DOMString>& classes() const;
            /**@}*/
        private:
            DOMString mTagName{};
            DOMString mId{};
            std::vector<DOMString> mClasses{};
            AttributeMap mAttributeMap{};
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
