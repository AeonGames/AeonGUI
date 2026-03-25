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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "aeongui/dom/Event.hpp"
#include "aeongui/dom/EventTarget.hpp"
#include "aeongui/dom/EventListener.hpp"
#include "aeongui/dom/Node.hpp"
#include "aeongui/dom/UIEvent.hpp"
#include "aeongui/dom/KeyboardEvent.hpp"
#include "aeongui/dom/MouseEvent.hpp"
#include "aeongui/dom/WheelEvent.hpp"
#include "aeongui/dom/FocusEvent.hpp"

namespace
{
    /// Minimal concrete EventListener that records calls.
    class RecordingListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            callCount++;
            lastType = event.type();
            lastPhase = event.eventPhase();
            lastTarget = event.target();
            lastCurrentTarget = event.currentTarget();
        }
        int callCount{0};
        AeonGUI::DOM::DOMString lastType;
        uint16_t lastPhase{0};
        const AeonGUI::DOM::EventTarget* lastTarget{nullptr};
        const AeonGUI::DOM::EventTarget* lastCurrentTarget{nullptr};
    };

    /// Listener that stops propagation.
    class StopPropagationListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            callCount++;
            event.stopPropagation();
        }
        int callCount{0};
    };

    /// Listener that stops immediate propagation.
    class StopImmediateListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            callCount++;
            event.stopImmediatePropagation();
        }
        int callCount{0};
    };

    /// Listener that calls preventDefault.
    class PreventDefaultListener : public AeonGUI::DOM::EventListener
    {
    public:
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            callCount++;
            event.preventDefault();
        }
        int callCount{0};
    };

    /// Minimal concrete Node for testing event dispatch through a tree.
    class TestNode : public AeonGUI::DOM::Node
    {
    public:
        explicit TestNode ( const std::string& aLabel, Node* aParent = nullptr )
            : Node ( aParent ), mLabel{aLabel} {}
        NodeType nodeType() const override
        {
            return ELEMENT_NODE;
        }
        const std::string& label() const
        {
            return mLabel;
        }
    private:
        std::string mLabel;
    };
}

// ---------- Event construction tests ----------

TEST ( EventTest, DefaultConstruction )
{
    AeonGUI::DOM::Event event ( "click" );
    EXPECT_EQ ( event.type(), "click" );
    EXPECT_FALSE ( event.bubbles() );
    EXPECT_FALSE ( event.cancelable() );
    EXPECT_FALSE ( event.composed() );
    EXPECT_FALSE ( event.defaultPrevented() );
    EXPECT_FALSE ( event.isTrusted() );
    EXPECT_EQ ( event.eventPhase(), event.NONE );
    EXPECT_EQ ( event.target(), nullptr );
    EXPECT_EQ ( event.currentTarget(), nullptr );
}

TEST ( EventTest, ConstructionWithInit )
{
    AeonGUI::DOM::Event event ( "mousedown", AeonGUI::DOM::EventInit{true, true, true} );
    EXPECT_EQ ( event.type(), "mousedown" );
    EXPECT_TRUE ( event.bubbles() );
    EXPECT_TRUE ( event.cancelable() );
    EXPECT_TRUE ( event.composed() );
}

TEST ( EventTest, PreventDefaultOnCancelable )
{
    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{false, true, false} );
    EXPECT_FALSE ( event.defaultPrevented() );
    event.preventDefault();
    EXPECT_TRUE ( event.defaultPrevented() );
}

TEST ( EventTest, PreventDefaultOnNonCancelable )
{
    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{false, false, false} );
    event.preventDefault();
    EXPECT_FALSE ( event.defaultPrevented() );
}

// ---------- EventTarget listener tests ----------

TEST ( EventTargetTest, AddAndDispatchListener )
{
    auto node = std::make_unique<TestNode> ( "target" );
    RecordingListener listener;
    node->addEventListener ( "click", &listener );

    AeonGUI::DOM::Event event ( "click" );
    node->dispatchEvent ( event );

    EXPECT_EQ ( listener.callCount, 1 );
    EXPECT_EQ ( listener.lastType, "click" );
    EXPECT_EQ ( listener.lastTarget, node.get() );
}

TEST ( EventTargetTest, DuplicateListenerNotAdded )
{
    auto node = std::make_unique<TestNode> ( "target" );
    RecordingListener listener;
    node->addEventListener ( "click", &listener );
    node->addEventListener ( "click", &listener ); // duplicate

    AeonGUI::DOM::Event event ( "click" );
    node->dispatchEvent ( event );

    EXPECT_EQ ( listener.callCount, 1 );
}

TEST ( EventTargetTest, RemoveListener )
{
    auto node = std::make_unique<TestNode> ( "target" );
    RecordingListener listener;
    node->addEventListener ( "click", &listener );
    node->removeEventListener ( "click", &listener );

    AeonGUI::DOM::Event event ( "click" );
    node->dispatchEvent ( event );

    EXPECT_EQ ( listener.callCount, 0 );
}

TEST ( EventTargetTest, OnceListenerRemovedAfterFirstCall )
{
    auto node = std::make_unique<TestNode> ( "target" );
    RecordingListener listener;
    AeonGUI::DOM::AddEventListenerOptions opts;
    opts.capture = false;
    opts.once = true;
    opts.passive = false;
    node->addEventListener ( "click", &listener, opts );

    AeonGUI::DOM::Event event1 ( "click" );
    node->dispatchEvent ( event1 );
    EXPECT_EQ ( listener.callCount, 1 );

    AeonGUI::DOM::Event event2 ( "click" );
    node->dispatchEvent ( event2 );
    EXPECT_EQ ( listener.callCount, 1 ); // not called again
}

TEST ( EventTargetTest, WrongTypeNotDispatched )
{
    auto node = std::make_unique<TestNode> ( "target" );
    RecordingListener listener;
    node->addEventListener ( "click", &listener );

    AeonGUI::DOM::Event event ( "mousemove" );
    node->dispatchEvent ( event );

    EXPECT_EQ ( listener.callCount, 0 );
}

// ---------- Event propagation tests ----------

TEST ( EventPropagationTest, BubblesUpTree )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener rootListener;
    RecordingListener childListener;
    root->addEventListener ( "click", &rootListener );
    child->addEventListener ( "click", &childListener );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    child->dispatchEvent ( event );

    EXPECT_EQ ( childListener.callCount, 1 );
    EXPECT_EQ ( rootListener.callCount, 1 );
}

TEST ( EventPropagationTest, NonBubblingDoesNotBubble )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener rootListener;
    RecordingListener childListener;
    root->addEventListener ( "focus", &rootListener );
    child->addEventListener ( "focus", &childListener );

    AeonGUI::DOM::Event event ( "focus", AeonGUI::DOM::EventInit{false, false, false} );
    child->dispatchEvent ( event );

    EXPECT_EQ ( childListener.callCount, 1 );
    EXPECT_EQ ( rootListener.callCount, 0 ); // non-bubbling: root not called in bubble phase
}

TEST ( EventPropagationTest, CaptureListenerCalledDuringCapture )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener captureListener;
    RecordingListener bubbleListener;
    root->addEventListener ( "click", &captureListener, true ); // capture
    root->addEventListener ( "click", &bubbleListener );        // bubble

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    child->dispatchEvent ( event );

    EXPECT_EQ ( captureListener.callCount, 1 );
    EXPECT_EQ ( captureListener.lastPhase, event.CAPTURING_PHASE );
    EXPECT_EQ ( bubbleListener.callCount, 1 );
    EXPECT_EQ ( bubbleListener.lastPhase, event.BUBBLING_PHASE );
}

TEST ( EventPropagationTest, StopPropagationPreventsParent )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener rootListener;
    StopPropagationListener childListener;
    root->addEventListener ( "click", &rootListener );
    child->addEventListener ( "click", &childListener );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    child->dispatchEvent ( event );

    EXPECT_EQ ( childListener.callCount, 1 );
    EXPECT_EQ ( rootListener.callCount, 0 ); // stopped
}

TEST ( EventPropagationTest, StopImmediatePreventsOtherListenersOnSameTarget )
{
    auto node = std::make_unique<TestNode> ( "target" );
    StopImmediateListener first;
    RecordingListener second;
    node->addEventListener ( "click", &first );
    node->addEventListener ( "click", &second );

    AeonGUI::DOM::Event event ( "click" );
    node->dispatchEvent ( event );

    EXPECT_EQ ( first.callCount, 1 );
    EXPECT_EQ ( second.callCount, 0 ); // immediate stop
}

TEST ( EventPropagationTest, DispatchReturnsFalseWhenDefaultPrevented )
{
    auto node = std::make_unique<TestNode> ( "target" );
    PreventDefaultListener listener;
    node->addEventListener ( "click", &listener );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{false, true, false} );
    bool result = node->dispatchEvent ( event );

    EXPECT_FALSE ( result );
    EXPECT_TRUE ( event.defaultPrevented() );
}

TEST ( EventPropagationTest, ComposedPathReflectsTree )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );
    auto* grandchild = static_cast<TestNode*> ( child->AddNode ( std::make_unique<TestNode> ( "grandchild", child ) ) );

    RecordingListener listener;
    grandchild->addEventListener ( "click", &listener );

    AeonGUI::DOM::Event event ( "click" );
    grandchild->dispatchEvent ( event );

    const auto& path = event.composedPath();
    ASSERT_EQ ( path.size(), 3u );
    EXPECT_EQ ( path[0], grandchild );
    EXPECT_EQ ( path[1], child );
    EXPECT_EQ ( path[2], root.get() );
}

// ---------- UIEvent tests ----------

TEST ( UIEventTest, DefaultConstruction )
{
    AeonGUI::DOM::UIEvent event ( "resize" );
    EXPECT_EQ ( event.type(), "resize" );
    EXPECT_EQ ( event.view(), nullptr );
    EXPECT_EQ ( event.detail(), 0 );
}

TEST ( UIEventTest, ConstructionWithInit )
{
    AeonGUI::DOM::UIEventInit init;
    init.bubbles = true;
    init.detail = 42;
    AeonGUI::DOM::UIEvent event ( "resize", init );
    EXPECT_TRUE ( event.bubbles() );
    EXPECT_EQ ( event.detail(), 42 );
    EXPECT_EQ ( event.view(), nullptr );
}

// ---------- KeyboardEvent tests ----------

TEST ( KeyboardEventTest, DefaultConstruction )
{
    AeonGUI::DOM::KeyboardEvent event ( "keydown" );
    EXPECT_EQ ( event.type(), "keydown" );
    EXPECT_EQ ( event.key(), "" );
    EXPECT_EQ ( event.code(), "" );
    EXPECT_EQ ( event.location(), AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_STANDARD );
    EXPECT_FALSE ( event.repeat() );
    EXPECT_FALSE ( event.isComposing() );
    EXPECT_FALSE ( event.ctrlKey() );
    EXPECT_FALSE ( event.shiftKey() );
    EXPECT_FALSE ( event.altKey() );
    EXPECT_FALSE ( event.metaKey() );
}

TEST ( KeyboardEventTest, ConstructionWithInit )
{
    AeonGUI::DOM::KeyboardEventInit init;
    init.bubbles = true;
    init.cancelable = true;
    init.key = "Enter";
    init.code = "Enter";
    init.location = AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_STANDARD;
    init.repeat = true;
    init.ctrlKey = true;
    init.shiftKey = true;
    AeonGUI::DOM::KeyboardEvent event ( "keydown", init );
    EXPECT_EQ ( event.key(), "Enter" );
    EXPECT_EQ ( event.code(), "Enter" );
    EXPECT_TRUE ( event.repeat() );
    EXPECT_TRUE ( event.ctrlKey() );
    EXPECT_TRUE ( event.shiftKey() );
    EXPECT_FALSE ( event.altKey() );
    EXPECT_TRUE ( event.bubbles() );
    EXPECT_TRUE ( event.cancelable() );
}

TEST ( KeyboardEventTest, GetModifierState )
{
    AeonGUI::DOM::KeyboardEventInit init;
    init.ctrlKey = true;
    init.altKey = true;
    init.modifierCapsLock = true;
    init.modifierNumLock = true;
    AeonGUI::DOM::KeyboardEvent event ( "keydown", init );

    EXPECT_TRUE ( event.getModifierState ( "Control" ) );
    EXPECT_TRUE ( event.getModifierState ( "Alt" ) );
    EXPECT_FALSE ( event.getModifierState ( "Shift" ) );
    EXPECT_FALSE ( event.getModifierState ( "Meta" ) );
    EXPECT_TRUE ( event.getModifierState ( "CapsLock" ) );
    EXPECT_TRUE ( event.getModifierState ( "NumLock" ) );
    EXPECT_FALSE ( event.getModifierState ( "ScrollLock" ) );
    EXPECT_FALSE ( event.getModifierState ( "NonExistent" ) );
}

TEST ( KeyboardEventTest, LocationConstants )
{
    EXPECT_EQ ( AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_STANDARD, 0x00u );
    EXPECT_EQ ( AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_LEFT, 0x01u );
    EXPECT_EQ ( AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_RIGHT, 0x02u );
    EXPECT_EQ ( AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_NUMPAD, 0x03u );
}

TEST ( KeyboardEventTest, LeftShiftLocation )
{
    AeonGUI::DOM::KeyboardEventInit init;
    init.key = "Shift";
    init.code = "ShiftLeft";
    init.location = AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_LEFT;
    init.shiftKey = true;
    AeonGUI::DOM::KeyboardEvent event ( "keydown", init );
    EXPECT_EQ ( event.location(), AeonGUI::DOM::KeyboardEvent::DOM_KEY_LOCATION_LEFT );
    EXPECT_TRUE ( event.shiftKey() );
}

// ---------- MouseEvent tests ----------

TEST ( MouseEventTest, DefaultConstruction )
{
    AeonGUI::DOM::MouseEvent event ( "click" );
    EXPECT_EQ ( event.type(), "click" );
    EXPECT_DOUBLE_EQ ( event.screenX(), 0.0 );
    EXPECT_DOUBLE_EQ ( event.screenY(), 0.0 );
    EXPECT_DOUBLE_EQ ( event.clientX(), 0.0 );
    EXPECT_DOUBLE_EQ ( event.clientY(), 0.0 );
    EXPECT_EQ ( event.button(), 0 );
    EXPECT_EQ ( event.buttons(), 0u );
    EXPECT_EQ ( event.relatedTarget(), nullptr );
    EXPECT_FALSE ( event.ctrlKey() );
    EXPECT_FALSE ( event.shiftKey() );
    EXPECT_FALSE ( event.altKey() );
    EXPECT_FALSE ( event.metaKey() );
}

TEST ( MouseEventTest, ConstructionWithInit )
{
    AeonGUI::DOM::MouseEventInit init;
    init.bubbles = true;
    init.cancelable = true;
    init.screenX = 100.5;
    init.screenY = 200.5;
    init.clientX = 50.5;
    init.clientY = 75.5;
    init.button = 2;
    init.buttons = 4;
    init.ctrlKey = true;
    init.metaKey = true;
    AeonGUI::DOM::MouseEvent event ( "mousedown", init );

    EXPECT_EQ ( event.type(), "mousedown" );
    EXPECT_DOUBLE_EQ ( event.screenX(), 100.5 );
    EXPECT_DOUBLE_EQ ( event.screenY(), 200.5 );
    EXPECT_DOUBLE_EQ ( event.clientX(), 50.5 );
    EXPECT_DOUBLE_EQ ( event.clientY(), 75.5 );
    EXPECT_EQ ( event.button(), 2 );
    EXPECT_EQ ( event.buttons(), 4u );
    EXPECT_TRUE ( event.ctrlKey() );
    EXPECT_TRUE ( event.metaKey() );
    EXPECT_FALSE ( event.shiftKey() );
    EXPECT_TRUE ( event.bubbles() );
}

TEST ( MouseEventTest, GetModifierState )
{
    AeonGUI::DOM::MouseEventInit init;
    init.shiftKey = true;
    init.modifierSuper = true;
    AeonGUI::DOM::MouseEvent event ( "click", init );

    EXPECT_TRUE ( event.getModifierState ( "Shift" ) );
    EXPECT_TRUE ( event.getModifierState ( "Super" ) );
    EXPECT_FALSE ( event.getModifierState ( "Control" ) );
    EXPECT_FALSE ( event.getModifierState ( "Alt" ) );
}

TEST ( MouseEventTest, RelatedTarget )
{
    auto target = std::make_unique<TestNode> ( "target" );
    auto related = std::make_unique<TestNode> ( "related" );
    AeonGUI::DOM::MouseEventInit init;
    init.relatedTarget = related.get();
    AeonGUI::DOM::MouseEvent event ( "mouseenter", init );
    EXPECT_EQ ( event.relatedTarget(), related.get() );
}

// ---------- WheelEvent tests ----------

TEST ( WheelEventTest, DefaultConstruction )
{
    AeonGUI::DOM::WheelEvent event ( "wheel" );
    EXPECT_EQ ( event.type(), "wheel" );
    EXPECT_DOUBLE_EQ ( event.deltaX(), 0.0 );
    EXPECT_DOUBLE_EQ ( event.deltaY(), 0.0 );
    EXPECT_DOUBLE_EQ ( event.deltaZ(), 0.0 );
    EXPECT_EQ ( event.deltaMode(), AeonGUI::DOM::WheelEvent::DOM_DELTA_PIXEL );
}

TEST ( WheelEventTest, ConstructionWithInit )
{
    AeonGUI::DOM::WheelEventInit init;
    init.bubbles = true;
    init.deltaX = 10.0;
    init.deltaY = -120.0;
    init.deltaZ = 0.0;
    init.deltaMode = AeonGUI::DOM::WheelEvent::DOM_DELTA_LINE;
    init.clientX = 100.0;
    init.clientY = 200.0;
    AeonGUI::DOM::WheelEvent event ( "wheel", init );

    EXPECT_DOUBLE_EQ ( event.deltaX(), 10.0 );
    EXPECT_DOUBLE_EQ ( event.deltaY(), -120.0 );
    EXPECT_DOUBLE_EQ ( event.deltaZ(), 0.0 );
    EXPECT_EQ ( event.deltaMode(), AeonGUI::DOM::WheelEvent::DOM_DELTA_LINE );
    EXPECT_DOUBLE_EQ ( event.clientX(), 100.0 );
    EXPECT_DOUBLE_EQ ( event.clientY(), 200.0 );
    EXPECT_TRUE ( event.bubbles() );
}

TEST ( WheelEventTest, DeltaModeConstants )
{
    EXPECT_EQ ( AeonGUI::DOM::WheelEvent::DOM_DELTA_PIXEL, 0x00u );
    EXPECT_EQ ( AeonGUI::DOM::WheelEvent::DOM_DELTA_LINE, 0x01u );
    EXPECT_EQ ( AeonGUI::DOM::WheelEvent::DOM_DELTA_PAGE, 0x02u );
}

TEST ( WheelEventTest, InheritsMouseEventProperties )
{
    AeonGUI::DOM::WheelEventInit init;
    init.button = 1;
    init.buttons = 2;
    init.ctrlKey = true;
    init.deltaY = -100.0;
    AeonGUI::DOM::WheelEvent event ( "wheel", init );

    EXPECT_EQ ( event.button(), 1 );
    EXPECT_EQ ( event.buttons(), 2u );
    EXPECT_TRUE ( event.ctrlKey() );
    EXPECT_DOUBLE_EQ ( event.deltaY(), -100.0 );
}

// ---------- FocusEvent tests ----------

TEST ( FocusEventTest, DefaultConstruction )
{
    AeonGUI::DOM::FocusEvent event ( "focus" );
    EXPECT_EQ ( event.type(), "focus" );
    EXPECT_EQ ( event.relatedTarget(), nullptr );
    EXPECT_FALSE ( event.bubbles() );
}

TEST ( FocusEventTest, ConstructionWithInit )
{
    auto related = std::make_unique<TestNode> ( "related" );
    AeonGUI::DOM::FocusEventInit init;
    init.bubbles = true;
    init.relatedTarget = related.get();
    AeonGUI::DOM::FocusEvent event ( "focusin", init );

    EXPECT_EQ ( event.type(), "focusin" );
    EXPECT_TRUE ( event.bubbles() );
    EXPECT_EQ ( event.relatedTarget(), related.get() );
}

TEST ( FocusEventTest, BlurEvent )
{
    AeonGUI::DOM::FocusEventInit init;
    init.bubbles = false;
    AeonGUI::DOM::FocusEvent event ( "blur", init );
    EXPECT_EQ ( event.type(), "blur" );
    EXPECT_FALSE ( event.bubbles() );
}

// ---------- Integration: dispatch typed events through tree ----------

TEST ( EventDispatchIntegration, KeyboardEventBubbles )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener rootListener;
    RecordingListener childListener;
    root->addEventListener ( "keydown", &rootListener );
    child->addEventListener ( "keydown", &childListener );

    AeonGUI::DOM::KeyboardEventInit init;
    init.bubbles = true;
    init.cancelable = true;
    init.key = "a";
    init.code = "KeyA";
    AeonGUI::DOM::KeyboardEvent event ( "keydown", init );
    child->dispatchEvent ( event );

    EXPECT_EQ ( childListener.callCount, 1 );
    EXPECT_EQ ( rootListener.callCount, 1 );
    EXPECT_EQ ( childListener.lastType, "keydown" );
}

TEST ( EventDispatchIntegration, MouseEventCaptureAndBubble )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    RecordingListener captureListener;
    RecordingListener bubbleListener;
    RecordingListener childListener;
    root->addEventListener ( "mousedown", &captureListener, true );
    root->addEventListener ( "mousedown", &bubbleListener );
    child->addEventListener ( "mousedown", &childListener );

    AeonGUI::DOM::MouseEventInit init;
    init.bubbles = true;
    init.clientX = 100.0;
    init.clientY = 200.0;
    init.button = 0;
    AeonGUI::DOM::MouseEvent event ( "mousedown", init );
    child->dispatchEvent ( event );

    // Capture fires first on root, then target, then bubble on root
    EXPECT_EQ ( captureListener.callCount, 1 );
    EXPECT_EQ ( captureListener.lastPhase, event.CAPTURING_PHASE );
    EXPECT_EQ ( childListener.callCount, 1 );
    EXPECT_EQ ( childListener.lastPhase, event.AT_TARGET );
    EXPECT_EQ ( bubbleListener.callCount, 1 );
    EXPECT_EQ ( bubbleListener.lastPhase, event.BUBBLING_PHASE );
}

TEST ( EventDispatchIntegration, StopPropagationInCapturePhase )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* child = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "child", root.get() ) ) );

    StopPropagationListener captureListener;
    RecordingListener childListener;
    root->addEventListener ( "click", &captureListener, true ); // capture on root
    child->addEventListener ( "click", &childListener );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    child->dispatchEvent ( event );

    EXPECT_EQ ( captureListener.callCount, 1 );
    EXPECT_EQ ( childListener.callCount, 0 ); // stopped during capture
}

TEST ( EventDispatchIntegration, AtTargetBothCaptureAndBubbleListenersCalled )
{
    auto node = std::make_unique<TestNode> ( "target" );

    RecordingListener captureListener;
    RecordingListener bubbleListener;
    node->addEventListener ( "click", &captureListener, true );
    node->addEventListener ( "click", &bubbleListener );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    node->dispatchEvent ( event );

    // At target, both capture and bubble listeners should fire
    EXPECT_EQ ( captureListener.callCount, 1 );
    EXPECT_EQ ( captureListener.lastPhase, event.AT_TARGET );
    EXPECT_EQ ( bubbleListener.callCount, 1 );
    EXPECT_EQ ( bubbleListener.lastPhase, event.AT_TARGET );
}

TEST ( EventDispatchIntegration, ThreeLevelPropagation )
{
    auto root = std::make_unique<TestNode> ( "root" );
    auto* middle = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "middle", root.get() ) ) );
    auto* leaf = static_cast<TestNode*> ( middle->AddNode ( std::make_unique<TestNode> ( "leaf", middle ) ) );

    std::vector<std::string> order;

    // Use separate listeners to track order
    struct OrderListener : public AeonGUI::DOM::EventListener
    {
        OrderListener ( std::vector<std::string>& aOrder, const std::string& aName ) : order ( aOrder ), name ( aName ) {}
        void handleEvent ( AeonGUI::DOM::Event& event ) override
        {
            order.push_back ( name + ":" + std::to_string ( event.eventPhase() ) );
        }
        std::vector<std::string>& order;
        std::string name;
    };

    OrderListener rootCapture ( order, "root" );
    OrderListener middleCapture ( order, "middle" );
    OrderListener leafTarget ( order, "leaf" );
    OrderListener middleBubble ( order, "middle" );
    OrderListener rootBubble ( order, "root" );

    root->addEventListener ( "click", &rootCapture, true );
    middle->addEventListener ( "click", &middleCapture, true );
    leaf->addEventListener ( "click", &leafTarget );
    middle->addEventListener ( "click", &middleBubble );
    root->addEventListener ( "click", &rootBubble );

    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, false, false} );
    leaf->dispatchEvent ( event );

    // Expected order: root capture (1), middle capture (1), leaf at-target (2), middle bubble (3), root bubble (3)
    ASSERT_EQ ( order.size(), 5u );
    EXPECT_EQ ( order[0], "root:1" );   // CAPTURING_PHASE
    EXPECT_EQ ( order[1], "middle:1" ); // CAPTURING_PHASE
    EXPECT_EQ ( order[2], "leaf:2" );   // AT_TARGET
    EXPECT_EQ ( order[3], "middle:3" ); // BUBBLING_PHASE
    EXPECT_EQ ( order[4], "root:3" );   // BUBBLING_PHASE
}
