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
            AEONGUI_DLL Element ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Virtual destructor. */
            AEONGUI_DLL virtual ~Element();
            /**DOM Properties and Methods @{*/
            /** @brief Get the node type (always ELEMENT_NODE).
             *  @return NodeType::ELEMENT_NODE. */
            NodeType nodeType() const final;
            /** @brief Get the tag name.
             *  @return The element's tag name.
             */
            AEONGUI_DLL const DOMString& tagName() const;
            /** @brief Get the element's ID attribute value.
             *  @return The ID string.
             */
            AEONGUI_DLL const DOMString& id() const;
            /** @brief Get the element's CSS class list.
             *  @return Vector of interned lwc_string class names.
             */
            const std::vector<lwc_string*>& classes() const;
            /** @brief Get the element's attribute map.
             *  @return Const reference to the AttributeMap.
             */
            AEONGUI_DLL const AttributeMap& attributes() const;
            /** @brief Get the value of a named attribute.
             *  @param aName Attribute name.
             *  @return Pointer to the value string, or nullptr if not found.
             */
            AEONGUI_DLL const DOMString* getAttribute ( const DOMString& aName ) const;
            /** @brief Set (or add) a named attribute.
             *  @param aName  Attribute name.
             *  @param aValue New value.
             */
            AEONGUI_DLL void setAttribute ( const DOMString& aName, const DOMString& aValue );
            /** @brief Check the :hover pseudo-class state.
             *  @return true if this element is hovered.
             */
            AEONGUI_DLL bool isHover() const;
            /** @brief Check the :active pseudo-class state.
             *  @return true if this element is active (mouse held down).
             */
            AEONGUI_DLL bool isActive() const;
            /** @brief Check the :focus pseudo-class state.
             *  @return true if this element has focus.
             */
            AEONGUI_DLL bool isFocus() const;
            /** @brief Set the :hover pseudo-class state.
             *  @param aHover true if the element is hovered. */
            void setHover ( bool aHover );
            /** @brief Set the :active pseudo-class state.
             *  @param aActive true if the element is active. */
            void setActive ( bool aActive );
            /** @brief Set the :focus pseudo-class state.
             *  @param aFocus true if the element has focus. */
            void setFocus ( bool aFocus );
            /** @brief Re-run CSS selection and compose with parent styles.
             *  @param aDocumentStyleSheet Optional document-level stylesheet.
             *  If provided, it is stored for future re-selections. */
            AEONGUI_DLL void ReselectCSS ( css_stylesheet* aDocumentStyleSheet = nullptr );
            /**@}*/
        private:
            DOMString mTagName{};
            DOMString mId{};
            std::vector<lwc_string*> mClasses{};
            AEONGUI_DLL void OnAncestorChanged() override;
            bool mIsHover{false};     ///< :hover pseudo-class state.
            bool mIsActive{false};    ///< :active pseudo-class state.
            bool mIsFocus{false};     ///< :focus pseudo-class state.
            css_stylesheet* mDocumentStyleSheet{nullptr}; ///< Non-owning pointer to document stylesheet.
        protected:
            AttributeMap mAttributes{}; ///< The element's attribute map.
            StyleSheetPtr mInlineStyleSheet{}; ///< Inline style parsed from the style attribute.
            SelectResultsPtr mComputedStyles{}; ///< Computed CSS styles for this element.
            /** @brief Get computed styles from the parent element.
             *  @return Pointer to parent's computed styles, or nullptr. */
            css_select_results* GetParentComputedStyles() const;
            /** @brief Get this element's computed styles.
             *  @return Pointer to computed styles, or nullptr. */
            css_select_results* GetComputedStyles() const;
            /** @brief Apply child transform animation overrides to the canvas.
             *  @param aCanvas The target canvas. */
            void ApplyChildTransformAnimations ( Canvas& aCanvas ) const;
            /** @brief Apply child paint animation overrides to the canvas.
             *  @param aCanvas The target canvas. */
            void ApplyChildPaintAnimations ( Canvas& aCanvas ) const;
            /** @brief Called after an attribute is changed via setAttribute.
             *
             *  The base implementation handles id, class, and style.
             *  Subclasses override to re-parse element-specific attributes
             *  (e.g. path data, rect dimensions) and rebuild internal state.
             *  @param aName  Attribute name that changed.
             *  @param aValue New attribute value.
             */
            virtual void onAttributeChanged ( const DOMString& aName, const DOMString& aValue );
        public:
            /** @brief Update the CSS viewport dimensions used for percentage resolution.
             *  @param aWidth  Viewport width in pixels.
             *  @param aHeight Viewport height in pixels.
             */
            AEONGUI_DLL static void UpdateCSSViewport ( uint32_t aWidth, uint32_t aHeight );
        };
    }
}
#endif
