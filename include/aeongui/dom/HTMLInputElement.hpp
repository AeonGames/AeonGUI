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
#ifndef AEONGUI_HTMLINPUTELEMENT_H
#define AEONGUI_HTMLINPUTELEMENT_H

#include <cstdint>
#include "HTMLFormControlElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief HTML &lt;input&gt; element.
         *  @see https://html.spec.whatwg.org/multipage/input.html#htmlinputelement
         *
         *  Covers the original set of native controls: single-line text
         *  fields, password fields, hidden fields, checkboxes, radio
         *  buttons, and the three button flavours.  `type="file"` is
         *  deliberately unsupported — a file picker needs filesystem
         *  privileges beyond what an embedded UI toolkit should assume.
         */
        class HTMLInputElement : public HTMLFormControlElement
        {
        public:
            /** @brief Normalized `type` attribute value. */
            enum class Type
            {
                Text,       ///< type="text" and the unknown-type fallback.
                Password,   ///< type="password", value rendered as bullets.
                Search,     ///< type="search", behaves as text.
                Tel,        ///< type="tel", behaves as text.
                Url,        ///< type="url", behaves as text.
                Email,      ///< type="email", behaves as text.
                Number,     ///< type="number", behaves as text.
                Range,      ///< type="range", a horizontal slider.
                Hidden,     ///< type="hidden", never rendered.
                Checkbox,   ///< type="checkbox".
                Radio,      ///< type="radio".
                Button,     ///< type="button".
                Submit,     ///< type="submit".
                Reset,      ///< type="reset".
                Unsupported ///< type="file"/"image": parsed but not rendered.
            };

            HTMLInputElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~HTMLInputElement() final;

            /** @brief Normalized control type. */
            AEONGUI_DLL Type type() const;
            /** @brief Value declared in the markup (`value` attribute). */
            AEONGUI_DLL DOMString defaultValue() const;
            /** @brief Checkedness declared in the markup (`checked` attribute). */
            AEONGUI_DLL bool defaultChecked() const;
            /** @brief Current checkedness of a checkbox or radio. */
            AEONGUI_DLL bool checked() const;
            /** @brief Set the checkedness, unchecking the radio group peers. */
            AEONGUI_DLL void setChecked ( bool aChecked );
            /** @brief Placeholder shown while a text field is empty. */
            AEONGUI_DLL DOMString placeholder() const;
            /** @brief Whether the value can be edited by the user. */
            AEONGUI_DLL bool readOnly() const;
            /** @brief Visible width in characters (`size`, default 20). */
            AEONGUI_DLL uint32_t size() const;
            /** @brief Maximum value length in code points, or -1 when absent. */
            AEONGUI_DLL int32_t maxLength() const;
            /** @brief Caret position as a UTF-8 byte offset into the value. */
            AEONGUI_DLL size_t caretPosition() const;
            /** @brief Lower bound of a range control (`min`, default 0). */
            AEONGUI_DLL double min() const;
            /** @brief Upper bound of a range control (`max`, default 100),
             *  never below min(). */
            AEONGUI_DLL double max() const;
            /** @brief Granularity of a range control (`step`, default 1).
             *  Returns 0 for `step="any"`, meaning no snapping. */
            AEONGUI_DLL double step() const;
            /** @brief Current value of a range control as a number,
             *  clamped to [min, max] and snapped to step(). */
            AEONGUI_DLL double valueAsNumber() const;
            /** @brief Set a range control's value from a number. */
            AEONGUI_DLL void setValueAsNumber ( double aValue );

            AEONGUI_DLL DOMString value() const override;
            AEONGUI_DLL void setValue ( const DOMString& aValue ) override;
            AEONGUI_DLL void Reset() override;
            AEONGUI_DLL void AppendFormData ( FormData& aFormData ) const override;
            AEONGUI_DLL void Activate() override;
            AEONGUI_DLL bool IsTextEditable() const override;
            AEONGUI_DLL bool HandleKey ( const DOMString& aKey ) override;
            AEONGUI_DLL bool HandlePointerDrag ( double aX, double aY ) override;
            AEONGUI_DLL void EndPointerDrag() override;
            AEONGUI_DLL bool isChecked() const override;

            AEONGUI_DLL void DrawStart ( Canvas& aCanvas ) const override;

            /** @brief Intrinsic content-box size in CSS pixels, used by
             *  HTMLLayoutEngine's measure callback. */
            AEONGUI_DLL bool GetIntrinsicContentSize ( float& aWidth, float& aHeight ) const override;

        protected:
            void onAttributeChanged ( const DOMString& aName, const DOMString& aValue ) override;

        private:
            /// Uncheck every other radio sharing this control's name
            /// within the same form (or the same tree when unowned).
            void ClearRadioGroup();
            /// UA label for a value-less submit/reset button.
            DOMString DefaultButtonLabel() const;
            /// Value as painted: bullets for password fields.
            DOMString RenderedValue() const;
            /// Caret byte offset translated into RenderedValue() space.
            size_t CaretOffsetInRenderedValue() const;
            void PaintRadio ( Canvas& aCanvas ) const;
            void PaintCheckMark ( Canvas& aCanvas ) const;
            void PaintButtonLabel ( Canvas& aCanvas ) const;
            void PaintTextField ( Canvas& aCanvas ) const;
            void PaintRange ( Canvas& aCanvas ) const;
            /// Horizontal extent the slider thumb centre can travel over,
            /// inset by the thumb radius at both ends.
            void RangeTrackGeometry ( double& aStartX, double& aTrackWidth, double& aThumbRadius ) const;
            /// Store a range value, snapping and clamping it first.
            /// @return true when the stored value actually changed.
            bool SetRangeValue ( double aValue );
            /// Fill a thick line segment as a quadrilateral; the check
            /// mark is two of these, and the canvas stroke state stays
            /// reserved for SVG geometry.
            static void FillStroke ( Canvas& aCanvas, double aX0, double aY0,
                                     double aX1, double aY1, double aThickness );

            Type mType{Type::Text};
            DOMString mValue{};
            size_t mCaret{0};
            bool mChecked{false};
            /// Set while a pointer drag has moved a range value, so the
            /// `change` event can fire once the drag ends.
            bool mRangeDirty{false};
        };
    }
}
#endif
