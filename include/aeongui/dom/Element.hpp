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
        /** @brief Base class for DOM elements.
         *
         *  An Element has a tag name, attributes, CSS classes, and
         *  computed styles. It extends Node with the DOM Element interface.
         *  @see https://dom.spec.whatwg.org/#interface-element
         */
        class Element : public Node
        {
        public:
            /** @brief Construct an element.
             *  @param aTagName      The tag name (e.g. "rect", "circle").
             *  @param aAttributes   The element's attribute map.
             *  @param aParent       The parent node.
             */
            DLL Element ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Virtual destructor. */
            DLL virtual ~Element();
            /**DOM Properties and Methods @{*/
            /** @brief Get the node type (always ELEMENT_NODE). */
            NodeType nodeType() const final;
            /** @brief Get the tag name.
             *  @return The element's tag name.
             */
            const DOMString& tagName() const;
            /** @brief Get the element's ID attribute value.
             *  @return The ID string.
             */
            const DOMString& id() const;
            /** @brief Get the element's CSS class list.
             *  @return Vector of interned lwc_string class names.
             */
            const std::vector<lwc_string*>& classes() const;
            /** @brief Get the element's attribute map.
             *  @return Const reference to the AttributeMap.
             */
            const AttributeMap& attributes() const;
            /**@}*/
        private:
            DOMString mTagName{};
            DOMString mId{};
            std::vector<lwc_string*> mClasses{};
            DLL void OnAncestorChanged() override;
        protected:
            AttributeMap mAttributes{};
            StyleSheetPtr mInlineStyleSheet{};
            SelectResultsPtr mComputedStyles{};
            css_select_results* GetParentComputedStyles() const;
            css_select_results* GetComputedStyles() const;
        };
    }
}
#endif
