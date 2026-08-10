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

// Native form control coverage: element construction, value/checkedness
// state, keyboard editing, form submission entry lists, and the layout
// boxes the widgets get from HTMLLayoutEngine.

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "aeongui/ElementFactory.hpp"
#include "aeongui/HTMLLayoutEngine.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/dom/EventListener.hpp"
#include "aeongui/dom/HTMLBodyElement.hpp"
#include "aeongui/dom/HTMLButtonElement.hpp"
#include "aeongui/dom/HTMLFieldSetElement.hpp"
#include "aeongui/dom/HTMLFormElement.hpp"
#include "aeongui/dom/HTMLInputElement.hpp"
#include "aeongui/dom/HTMLLabelElement.hpp"
#include "aeongui/dom/HTMLTextAreaElement.hpp"
#include "aeongui/dom/Location.hpp"
#include "aeongui/dom/Text.hpp"
#include "aeongui/dom/Window.hpp"

namespace
{
    constexpr const char* kXHTML = "http://www.w3.org/1999/xhtml";

    using AeonGUI::AttributeMap;
    using AeonGUI::DOM::Node;

    /// Scoped temporary XHTML file, removed on destruction.
    class TempXHTML
    {
    public:
        explicit TempXHTML ( const std::string& aContent )
            : mPath{ std::filesystem::temp_directory_path() / "aeongui-form-controls.xhtml" }
        {
            std::ofstream file ( mPath, std::ios::binary | std::ios::out );
            file << aContent;
        }
        ~TempXHTML()
        {
            std::error_code ec;
            std::filesystem::remove ( mPath, ec );
        }
        std::string path() const
        {
            return mPath.generic_string();
        }
    private:
        std::filesystem::path mPath;
    };

    /// Construct an XHTML element through the factory and downcast it.
    template<class T>
    std::unique_ptr<T> Make ( const char* aTag, AttributeMap&& aAttributes, Node* aParent )
    {
        auto element = AeonGUI::Construct ( kXHTML, aTag, std::move ( aAttributes ), aParent );
        T* typed = dynamic_cast<T*> ( element.get() );
        EXPECT_NE ( typed, nullptr ) << "factory did not produce the expected type for <" << aTag << ">";
        element.release();
        return std::unique_ptr<T> ( typed );
    }

    template<class T>
    T* Attach ( Node* aParent, std::unique_ptr<T> aChild )
    {
        T* raw = aChild.get();
        aParent->AddNode ( std::move ( aChild ) );
        return raw;
    }

    /// Both canvas backends expose pixels as BGRA8; returned 0xAARRGGBB.
    uint32_t SamplePixel ( const uint8_t* aPixels, size_t aStride, int aX, int aY )
    {
        const uint8_t* p = aPixels + ( aY * aStride ) + ( aX * 4 );
        return ( static_cast<uint32_t> ( p[3] ) << 24 ) |
               ( static_cast<uint32_t> ( p[2] ) << 16 ) |
               ( static_cast<uint32_t> ( p[1] ) <<  8 ) |
               static_cast<uint32_t> ( p[0] );
    }

    /// Records how many times an event of the registered type fired.
    class CountingListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& ) override
        {
            ++count;
        }
        int count{0};
    };
}

TEST ( HTMLFormControls, FactoryProducesFormElementTypes )
{
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLFormElement*> (
                    AeonGUI::Construct ( kXHTML, "form", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                    AeonGUI::Construct ( kXHTML, "input", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLButtonElement*> (
                    AeonGUI::Construct ( kXHTML, "button", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLTextAreaElement*> (
                    AeonGUI::Construct ( kXHTML, "textarea", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLLabelElement*> (
                    AeonGUI::Construct ( kXHTML, "label", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLFieldSetElement*> (
                    AeonGUI::Construct ( kXHTML, "fieldset", {}, nullptr ).get() ), nullptr );
    EXPECT_NE ( dynamic_cast<AeonGUI::DOM::HTMLLegendElement*> (
                    AeonGUI::Construct ( kXHTML, "legend", {}, nullptr ).get() ), nullptr );
}

TEST ( HTMLFormControls, InputTypeParsingFallsBackToText )
{
    using Type = AeonGUI::DOM::HTMLInputElement::Type;
    EXPECT_EQ ( Make<AeonGUI::DOM::HTMLInputElement> ( "input", {}, nullptr )->type(), Type::Text );
    EXPECT_EQ ( Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "CheckBox"} }, nullptr )->type(), Type::Checkbox );
    EXPECT_EQ ( Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "password"} }, nullptr )->type(), Type::Password );
    EXPECT_EQ ( Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "color"} }, nullptr )->type(), Type::Text );
    // File pickers need privileges beyond the embedding app, so they
    // are parsed but never rendered or submitted.
    EXPECT_EQ ( Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "file"} }, nullptr )->type(), Type::Unsupported );
}

TEST ( HTMLFormControls, TextFieldEditing )
{
    auto input = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "text"}, {"value", "ab"} }, nullptr );

    EXPECT_EQ ( input->value(), "ab" );
    EXPECT_EQ ( input->caretPosition(), 2u );

    EXPECT_TRUE ( input->HandleKey ( "c" ) );
    EXPECT_EQ ( input->value(), "abc" );

    EXPECT_TRUE ( input->HandleKey ( "Backspace" ) );
    EXPECT_EQ ( input->value(), "ab" );

    EXPECT_TRUE ( input->HandleKey ( "ArrowLeft" ) );
    EXPECT_TRUE ( input->HandleKey ( "X" ) );
    EXPECT_EQ ( input->value(), "aXb" );

    EXPECT_TRUE ( input->HandleKey ( "Home" ) );
    EXPECT_TRUE ( input->HandleKey ( "Delete" ) );
    EXPECT_EQ ( input->value(), "Xb" );

    // Named keys that the control does not understand are left for
    // the embedder to deal with.
    EXPECT_FALSE ( input->HandleKey ( "F5" ) );
}

TEST ( HTMLFormControls, TextFieldRespectsMaxLengthAndReadOnly )
{
    auto limited = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"value", "abc"}, {"maxlength", "3"} }, nullptr );
    limited->HandleKey ( "d" );
    EXPECT_EQ ( limited->value(), "abc" );

    auto frozen = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"value", "abc"}, {"readonly", "readonly"} }, nullptr );
    EXPECT_FALSE ( frozen->IsTextEditable() );
    frozen->HandleKey ( "d" );
    EXPECT_EQ ( frozen->value(), "abc" );
}

TEST ( HTMLFormControls, CheckboxTogglesOnActivation )
{
    auto checkbox = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"}, {"name", "v"} }, nullptr );
    EXPECT_FALSE ( checkbox->checked() );
    EXPECT_FALSE ( checkbox->isChecked() );

    checkbox->Activate();
    EXPECT_TRUE ( checkbox->checked() );
    EXPECT_TRUE ( checkbox->isChecked() );

    checkbox->Activate();
    EXPECT_FALSE ( checkbox->checked() );

    // Space is the keyboard equivalent of a click.
    EXPECT_TRUE ( checkbox->HandleKey ( " " ) );
    EXPECT_TRUE ( checkbox->checked() );
}

TEST ( HTMLFormControls, RadioGroupIsMutuallyExclusive )
{
    auto form = Make<AeonGUI::DOM::HTMLFormElement> ( "form", {}, nullptr );
    auto* soup = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "radio"}, {"name", "meal"}, {"value", "soup"}, {"checked", "checked"} }, form.get() ) );
    auto* curry = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "radio"}, {"name", "meal"}, {"value", "curry"} }, form.get() ) );
    auto* other = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "radio"}, {"name", "drink"}, {"value", "tea"}, {"checked", "checked"} }, form.get() ) );

    EXPECT_TRUE ( soup->checked() );

    curry->Activate();
    EXPECT_TRUE ( curry->checked() );
    EXPECT_FALSE ( soup->checked() );
    // A differently-named radio is a different group.
    EXPECT_TRUE ( other->checked() );

    // Activating an already-checked radio never unchecks it.
    curry->Activate();
    EXPECT_TRUE ( curry->checked() );
}

TEST ( HTMLFormControls, FormDataSkipsUnsuccessfulControls )
{
    auto form = Make<AeonGUI::DOM::HTMLFormElement> ( "form", {}, nullptr );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"name", "comment"}, {"value", "hi"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"}, {"name", "on"}, {"checked", "checked"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"}, {"name", "off"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"name", "nope"}, {"value", "x"}, {"disabled", "disabled"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"value", "unnamed"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "submit"}, {"name", "go"}, {"value", "Send"} }, form.get() ) );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "hidden"}, {"name", "stamp"}, {"value", "1286705410"} }, form.get() ) );

    const auto data = form->GetFormData();
    ASSERT_EQ ( data.size(), 3u );
    EXPECT_EQ ( data[0].first,  "comment" );
    EXPECT_EQ ( data[0].second, "hi" );
    // A checked checkbox without a value attribute submits "on".
    EXPECT_EQ ( data[1].first,  "on" );
    EXPECT_EQ ( data[1].second, "on" );
    EXPECT_EQ ( data[2].first,  "stamp" );
    EXPECT_EQ ( data[2].second, "1286705410" );
}

TEST ( HTMLFormControls, SubmitAndResetButtonsDriveTheForm )
{
    auto form = Make<AeonGUI::DOM::HTMLFormElement> ( "form", {}, nullptr );
    auto* text = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"name", "comment"}, {"value", "default"} }, form.get() ) );
    auto* box = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"}, {"name", "v"}, {"checked", "checked"} }, form.get() ) );
    auto* submit = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "submit"}, {"value", "Send"} }, form.get() ) );
    auto* reset = Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "reset"}, {"value", "Reset"} }, form.get() ) );

    CountingListener submitted;
    form->addEventListener ( "submit", &submitted );

    text->setValue ( "edited" );
    box->Activate();
    EXPECT_EQ ( text->value(), "edited" );
    EXPECT_FALSE ( box->checked() );

    submit->Activate();
    EXPECT_EQ ( submitted.count, 1 );

    reset->Activate();
    EXPECT_EQ ( text->value(), "default" );
    EXPECT_TRUE ( box->checked() );
}

TEST ( HTMLFormControls, ButtonTypeDefaultsToSubmit )
{
    using Type = AeonGUI::DOM::HTMLButtonElement::Type;
    auto form = Make<AeonGUI::DOM::HTMLFormElement> ( "form", {}, nullptr );
    auto* button = Attach ( form.get(), Make<AeonGUI::DOM::HTMLButtonElement> ( "button", {}, form.get() ) );
    auto* plain = Attach ( form.get(), Make<AeonGUI::DOM::HTMLButtonElement> (
    "button", { {"type", "button"} }, form.get() ) );
    EXPECT_EQ ( button->type(), Type::Submit );
    EXPECT_EQ ( plain->type(), Type::Button );

    CountingListener submitted;
    form->addEventListener ( "submit", &submitted );

    plain->Activate();
    EXPECT_EQ ( submitted.count, 0 );

    button->Activate();
    EXPECT_EQ ( submitted.count, 1 );
}

TEST ( HTMLFormControls, DisabledFieldSetDisablesDescendants )
{
    auto fieldset = Make<AeonGUI::DOM::HTMLFieldSetElement> (
    "fieldset", { {"disabled", "disabled"} }, nullptr );
    auto* input = Attach ( fieldset.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"name", "x"}, {"value", "y"} }, fieldset.get() ) );

    EXPECT_TRUE ( input->isDisabled() );
    EXPECT_TRUE ( input->canBeDisabled() );

    AeonGUI::DOM::HTMLInputElement::FormData data;
    input->AppendFormData ( data );
    EXPECT_TRUE ( data.empty() );
}

TEST ( HTMLFormControls, LabelResolvesNestedControl )
{
    auto label = Make<AeonGUI::DOM::HTMLLabelElement> ( "label", {}, nullptr );
    auto* input = Attach ( label.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"} }, label.get() ) );
    EXPECT_EQ ( label->control(), input );
    EXPECT_EQ ( label->htmlFor(), "" );
}

TEST ( HTMLFormControls, TextAreaUsesTextContentAsDefaultValue )
{
    auto textarea = Make<AeonGUI::DOM::HTMLTextAreaElement> (
    "textarea", { {"rows", "3"}, {"cols", "10"} }, nullptr );
    textarea->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "hello", textarea.get() ) );

    EXPECT_EQ ( textarea->rows(), 3u );
    EXPECT_EQ ( textarea->cols(), 10u );
    EXPECT_EQ ( textarea->value(), "hello" );

    EXPECT_TRUE ( textarea->HandleKey ( "!" ) );
    EXPECT_EQ ( textarea->value(), "hello!" );

    EXPECT_TRUE ( textarea->HandleKey ( "Enter" ) );
    EXPECT_EQ ( textarea->value(), "hello!\n" );

    textarea->Reset();
    EXPECT_EQ ( textarea->value(), "hello" );
}

TEST ( HTMLFormControls, LayoutGivesControlsIntrinsicBoxes )
{
    auto body = Make<AeonGUI::DOM::HTMLBodyElement> ( "body", {}, nullptr );
    auto* text = Attach ( body.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "text"}, {"size", "10"} }, body.get() ) );
    auto* checkbox = Attach ( body.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "checkbox"} }, body.get() ) );
    auto* hidden = Attach ( body.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "hidden"} }, body.get() ) );

    AeonGUI::HTMLLayoutEngine engine;
    engine.Layout ( body.get(), 400.0f, 300.0f );

    // A text field is inline-level: it takes its intrinsic width, not
    // the full container width.
    EXPECT_GT ( text->GetLayoutBox().width, 0.0f );
    EXPECT_LT ( text->GetLayoutBox().width, 400.0f );
    EXPECT_GT ( text->GetLayoutBox().height, 0.0f );

    // The UA stylesheet pins checkboxes to 13x13.  Yoga sizes the
    // border box, so the 1px border eats into the content area.
    EXPECT_FLOAT_EQ ( checkbox->GetLayoutBox().width,  13.0f );
    EXPECT_FLOAT_EQ ( checkbox->GetLayoutBox().height, 13.0f );
    EXPECT_FLOAT_EQ ( checkbox->GetLayoutBox().contentWidth,  11.0f );

    // display: none collapses the hidden input entirely.
    EXPECT_FLOAT_EQ ( hidden->GetLayoutBox().width,  0.0f );
    EXPECT_FLOAT_EQ ( hidden->GetLayoutBox().height, 0.0f );
}

TEST ( HTMLFormControls, RangeSanitizesItsValue )
{
    using Type = AeonGUI::DOM::HTMLInputElement::Type;
    auto plain = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"} }, nullptr );
    EXPECT_EQ ( plain->type(), Type::Range );
    EXPECT_DOUBLE_EQ ( plain->min(), 0.0 );
    EXPECT_DOUBLE_EQ ( plain->max(), 100.0 );
    EXPECT_DOUBLE_EQ ( plain->step(), 1.0 );
    // Missing value defaults to the midpoint of the range.
    EXPECT_DOUBLE_EQ ( plain->valueAsNumber(), 50.0 );
    EXPECT_EQ ( plain->value(), "50" );

    auto stepped = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"}, {"min", "0"}, {"max", "10"},
        {"step", "2"}, {"value", "7"}
    }, nullptr );
    // 7 snaps to the nearest 0 + n*2.
    EXPECT_DOUBLE_EQ ( stepped->valueAsNumber(), 8.0 );

    stepped->setValueAsNumber ( 999.0 );
    EXPECT_DOUBLE_EQ ( stepped->valueAsNumber(), 10.0 );
    stepped->setValueAsNumber ( -5.0 );
    EXPECT_DOUBLE_EQ ( stepped->valueAsNumber(), 0.0 );

    auto any = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"}, {"step", "any"}, {"value", "12.5"} }, nullptr );
    EXPECT_DOUBLE_EQ ( any->step(), 0.0 );
    EXPECT_DOUBLE_EQ ( any->valueAsNumber(), 12.5 );
    EXPECT_EQ ( any->value(), "12.5" );

    // A max below min collapses the range instead of reversing it.
    auto inverted = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"}, {"min", "10"}, {"max", "2"} }, nullptr );
    EXPECT_DOUBLE_EQ ( inverted->max(), 10.0 );
    EXPECT_DOUBLE_EQ ( inverted->valueAsNumber(), 10.0 );
}

TEST ( HTMLFormControls, RangeRespondsToArrowKeys )
{
    auto slider = Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"}, {"min", "0"}, {"max", "10"},
        {"step", "2"}, {"value", "4"}
    }, nullptr );

    EXPECT_TRUE ( slider->HandleKey ( "ArrowRight" ) );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 6.0 );
    EXPECT_TRUE ( slider->HandleKey ( "ArrowDown" ) );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 4.0 );
    EXPECT_TRUE ( slider->HandleKey ( "Home" ) );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 0.0 );
    EXPECT_TRUE ( slider->HandleKey ( "End" ) );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 10.0 );
    // Stepping past the end clamps rather than wrapping.
    EXPECT_TRUE ( slider->HandleKey ( "ArrowRight" ) );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 10.0 );

    EXPECT_FALSE ( slider->HandleKey ( "a" ) );
    EXPECT_FALSE ( slider->IsTextEditable() );
}

TEST ( HTMLFormControls, RangeSubmitsItsValue )
{
    auto form = Make<AeonGUI::DOM::HTMLFormElement> ( "form", {}, nullptr );
    Attach ( form.get(), Make<AeonGUI::DOM::HTMLInputElement> (
    "input", { {"type", "range"}, {"name", "volume"}, {"max", "10"}, {"value", "3"} }, form.get() ) );

    const auto data = form->GetFormData();
    ASSERT_EQ ( data.size(), 1u );
    EXPECT_EQ ( data[0].first,  "volume" );
    EXPECT_EQ ( data[0].second, "3" );
}

TEST ( HTMLFormControls, DraggingARangeThroughTheWindowMovesTheThumb )
{
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <form>
      <input type="range" id="volume" name="volume" min="0" max="100" value="0"/>
    </form>
  </body>
</html>)XHTML"
    };

    AeonGUI::DOM::Window window ( 300u, 200u );
    window.location() = doc.path();
    window.Draw();

    auto* slider = dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                       window.document()->querySelector ( "input" ) );
    ASSERT_NE ( slider, nullptr );
    ASSERT_DOUBLE_EQ ( slider->valueAsNumber(), 0.0 );

    const auto& box = slider->GetLayoutBox();
    ASSERT_GT ( box.width,  0.0f );
    const double y = box.y + box.height * 0.5;

    // Press at the far right: the widget is clickable across its whole
    // box, not just where the thumb currently sits.
    window.HandleMouseDown ( box.x + box.width - 1.0, y );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 100.0 );

    // Drag back to the middle while still holding the button.
    window.HandleMouseMove ( box.x + box.width * 0.5, y );
    EXPECT_GT ( slider->valueAsNumber(), 30.0 );
    EXPECT_LT ( slider->valueAsNumber(), 70.0 );

    // Dragging past the left edge clamps at min instead of wrapping.
    window.HandleMouseMove ( box.x - 200.0, y );
    EXPECT_DOUBLE_EQ ( slider->valueAsNumber(), 0.0 );

    window.HandleMouseUp ( box.x - 200.0, y );
}

TEST ( HTMLFormControls, PressingAButtonRepaintsItsWholeBox )
{
    // Regression: pick bounds used to keep only the last path drawn
    // under a pick id, so a control that paints a background plus four
    // border edges reported the right border sliver as its bounds.
    // Partial redraws then clipped to that sliver and the :active
    // chrome never reached the screen until something forced a full
    // redraw.
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <form>
      <input type="submit" id="go" value="Send"/>
    </form>
  </body>
</html>)XHTML"
    };

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = doc.path();
    window.Draw();

    auto* button = dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                       window.document()->querySelector ( "input" ) );
    ASSERT_NE ( button, nullptr );

    const auto& box = button->GetLayoutBox();
    ASSERT_GT ( box.width,  12.0f );
    ASSERT_GT ( box.height, 6.0f );

    // Inside the border but left of the centered label, so only the
    // background colour is under this pixel.
    const int sample_x = static_cast<int> ( box.x ) + 3;
    const int sample_y = static_cast<int> ( box.y + box.height * 0.5f );
    const uint32_t idle = SamplePixel ( window.GetPixels(), window.GetStride(),
                                        sample_x, sample_y );

    window.HandleMouseDown ( box.x + box.width * 0.5, box.y + box.height * 0.5 );
    ASSERT_TRUE ( window.Draw() ) << "pressing the button should dirty the document";

    const uint32_t pressed = SamplePixel ( window.GetPixels(), window.GetStride(),
                                           sample_x, sample_y );
    EXPECT_NE ( pressed, idle )
            << "the :active background must repaint on press, not only after release";

    window.HandleMouseUp ( box.x + box.width * 0.5, box.y + box.height * 0.5 );
    ASSERT_TRUE ( window.Draw() );
    EXPECT_NE ( SamplePixel ( window.GetPixels(), window.GetStride(), sample_x, sample_y ),
                pressed )
            << "releasing must drop the :active background again";
}

TEST ( HTMLFormControls, ClickingALabelActivatesItsControl )
{
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <form>
      <input type="checkbox" id="peas" name="vegetable" value="peas"/>
      <label for="peas">Peas</label>
    </form>
  </body>
</html>)XHTML"
    };

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = doc.path();
    window.Draw();

    auto* checkbox = dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                         window.document()->querySelector ( "input" ) );
    auto* label = dynamic_cast<AeonGUI::DOM::HTMLLabelElement*> (
                      window.document()->querySelector ( "label" ) );
    ASSERT_NE ( checkbox, nullptr );
    ASSERT_NE ( label, nullptr );
    ASSERT_EQ ( label->control(), checkbox );
    ASSERT_FALSE ( checkbox->checked() );

    // A label paints text only, which does not stamp the pick buffer on
    // its own; the label has to contribute a hit area to be clickable.
    const auto& box = label->GetLayoutBox();
    ASSERT_GT ( box.width,  0.0f );
    const double x = box.x + box.width  * 0.5;
    const double y = box.y + box.height * 0.5;

    window.HandleMouseDown ( x, y );
    window.HandleMouseUp ( x, y );

    EXPECT_TRUE ( checkbox->checked() ) << "clicking a label should activate its control";
}

TEST ( HTMLFormControls, ClickingThroughTheWindowTogglesACheckbox )
{
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <form>
      <input type="checkbox" id="box" name="v"/>
    </form>
  </body>
</html>)XHTML"
    };

    AeonGUI::DOM::Window window ( 200u, 200u );
    window.location() = doc.path();
    window.Draw();

    auto* checkbox = dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                         window.document()->querySelector ( "input" ) );
    ASSERT_NE ( checkbox, nullptr );
    ASSERT_FALSE ( checkbox->checked() );

    const auto& box = checkbox->GetLayoutBox();
    ASSERT_GT ( box.width,  0.0f );
    ASSERT_GT ( box.height, 0.0f );
    const double x = box.x + box.width  * 0.5;
    const double y = box.y + box.height * 0.5;

    window.HandleMouseDown ( x, y );
    window.HandleMouseUp ( x, y );

    EXPECT_TRUE ( checkbox->checked() ) << "clicking the widget should hit-test to the input";
    EXPECT_TRUE ( checkbox->isFocus() );
}

TEST ( HTMLFormControls, TypingIntoAFocusedTextFieldEditsTheValue )
{
    TempXHTML doc
    {
        R"XHTML(<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <form>
      <input type="text" id="comment" name="comment" value="ab"/>
    </form>
  </body>
</html>)XHTML"
    };

    AeonGUI::DOM::Window window ( 300u, 200u );
    window.location() = doc.path();
    window.Draw();

    auto* input = dynamic_cast<AeonGUI::DOM::HTMLInputElement*> (
                      window.document()->querySelector ( "input" ) );
    ASSERT_NE ( input, nullptr );

    const auto& box = input->GetLayoutBox();
    window.HandleMouseDown ( box.x + box.width * 0.5, box.y + box.height * 0.5 );
    window.HandleMouseUp ( box.x + box.width * 0.5, box.y + box.height * 0.5 );
    ASSERT_TRUE ( input->isFocus() );

    window.HandleKeyDown ( "c", "KeyC" );
    EXPECT_EQ ( input->value(), "abc" );

    window.HandleKeyDown ( "Backspace", "Backspace" );
    EXPECT_EQ ( input->value(), "ab" );
}
