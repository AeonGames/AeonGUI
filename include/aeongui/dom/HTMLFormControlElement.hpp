/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_HTMLFORMCONTROLELEMENT_H
#define AEONGUI_HTMLFORMCONTROLELEMENT_H

#include <utility>
#include <vector>
#include "HTMLElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class HTMLFormElement;

        /** @brief Shared base for the native form controls
         *  (&lt;input&gt;, &lt;button&gt;, &lt;textarea&gt;).
         *
         *  Not a spec interface — the HTML standard repeats these members
         *  on every concrete control.  Gathering them here keeps form-owner
         *  lookup, the disabled state, activation behavior, and the
         *  submission entry list in one place, and gives the layout,
         *  paint, and hit-test passes a single type to dispatch on.
         *
         *  @see https://html.spec.whatwg.org/multipage/forms.html#form-associated-element
         */
        class HTMLFormControlElement : public HTMLElement
        {
        public:
            /** @brief Submission entry list: ordered name/value pairs. */
            using FormData = std::vector<std::pair<DOMString, DOMString>>;

            HTMLFormControlElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLFormControlElement() override;

            /** @brief The control's submission name (`name` attribute). */
            AEONGUI_DLL DOMString name() const;
            /** @brief The owning form: the element referenced by the `form`
             *  attribute if present, otherwise the nearest &lt;form&gt;
             *  ancestor.  nullptr when the control is not form-associated. */
            AEONGUI_DLL HTMLFormElement* form() const;
            /** @brief Current value of the control. */
            AEONGUI_DLL virtual DOMString value() const;
            /** @brief Replace the current value. */
            AEONGUI_DLL virtual void setValue ( const DOMString& aValue );
            /** @brief Restore the value/checkedness declared in the markup. */
            AEONGUI_DLL virtual void Reset();
            /** @brief Append this control's successful entries to @p aFormData.
             *
             *  Disabled and unnamed controls contribute nothing, matching
             *  the "constructing the entry list" algorithm. */
            AEONGUI_DLL virtual void AppendFormData ( FormData& aFormData ) const;
            /** @brief Run the control's activation behavior, i.e. what a
             *  click does once the `click` event goes unprevented. */
            AEONGUI_DLL virtual void Activate();
            /** @brief Whether the control edits text on key input. */
            AEONGUI_DLL virtual bool IsTextEditable() const;
            /** @brief Apply a keydown to a focused control.
             *  @param aKey DOM `KeyboardEvent.key` value.
             *  @return true when the key was consumed by the control. */
            AEONGUI_DLL virtual bool HandleKey ( const DOMString& aKey );
            /** @brief Track a pointer press or drag in document coordinates.
             *
             *  Called on mousedown and then on every mousemove while the
             *  control stays the active element, so a slider keeps
             *  following the pointer even once it leaves the widget.
             *  @return true when the control consumed the position. */
            AEONGUI_DLL virtual bool HandlePointerDrag ( double aX, double aY );
            /** @brief Called when a captured pointer drag ends, wherever the
             *  pointer happens to be by then. */
            AEONGUI_DLL virtual void EndPointerDrag();
            /** @brief Intrinsic content-box size in CSS pixels, for controls
             *  that render their own widget chrome instead of laying out
             *  child text.
             *  @param aWidth  Receives the intrinsic content width.
             *  @param aHeight Receives the intrinsic content height.
             *  @return false when the control has no intrinsic size and
             *          should be sized from its children, as &lt;button&gt; is. */
            AEONGUI_DLL virtual bool GetIntrinsicContentSize ( float& aWidth, float& aHeight ) const;

            AEONGUI_DLL bool canBeDisabled() const override;
            AEONGUI_DLL bool isDisabled() const override;

        protected:
            /** @brief Font parameters resolved from the computed style. */
            struct FontSpec
            {
                DOMString family{"sans-serif"}; ///< CSS font-family.
                double size{16.0};              ///< Font size in CSS pixels.
                int weight{400};                ///< CSS numeric font weight.
                int style{0};                   ///< 0 normal, 1 italic, 2 oblique.
            };
            /** @brief Read the computed font properties of this control. */
            FontSpec GetFontSpec() const;
            /** @brief Paint a 1px text caret using the computed `color`. */
            void PaintCaret ( Canvas& aCanvas, double aX, double aTop, double aHeight ) const;
            /** @brief Mark the owner document dirty so a state change
             *  (checkedness, value, caret) reaches the screen. */
            void RequestRedraw() const;
        };
    }
}
#endif
