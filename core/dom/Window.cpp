/*
Copyright (C) 2019,2020,2023,2025,2026 Rodrigo Jose Hernandez Cordoba

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

#include <iostream>
#include <stdexcept>
#include <string>
#include <libxml/tree.h>
#include <libxml/parser.h>

#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/MouseEvent.hpp"
#include "aeongui/dom/KeyboardEvent.hpp"
#include "aeongui/dom/WheelEvent.hpp"
#include "aeongui/dom/FocusEvent.hpp"
#include <vector>
#include <algorithm>

namespace AeonGUI
{
    namespace DOM
    {
        Window::Window () = default;
        Window::Window ( uint32_t aWidth, uint32_t aHeight ) :
            mCanvas{aWidth, aHeight}
        {
        }

        Window::~Window() = default;

        const Document* Window::document() const
        {
            return &mDocument;
        }

        Location& Window::location() const
        {
            return const_cast<Location&> ( mLocation );
        }

        void Window::OnLocationChanged ( const Location& location )
        {
            mDocument.Load ( location.href() );
        }

        void Window::ResizeViewport ( uint32_t aWidth, uint32_t aHeight )
        {
            mCanvas.ResizeViewport ( aWidth, aHeight );
        }

        const uint8_t* Window::GetPixels() const
        {
            return mCanvas.GetPixels();
        }

        size_t Window::GetWidth() const
        {
            return mCanvas.GetWidth();
        }
        size_t Window::GetHeight() const
        {
            return mCanvas.GetHeight();
        }
        size_t Window::GetStride() const
        {
            return mCanvas.GetStride();
        }

        void Window::Draw()
        {
            mCanvas.Clear();
            mDocument.Draw ( mCanvas );
        }

        void Window::Update ( double aDeltaTime )
        {
            mDocument.AdvanceTime ( aDeltaTime );
        }

        void Window::HandleMouseMove ( double aX, double aY, unsigned short aButtons,
                                       bool aCtrlKey, bool aShiftKey,
                                       bool aAltKey, bool aMetaKey )
        {
            Element* target = mDocument.elementFromPoint ( mCanvas, aX, aY );
            // Handle mouseenter/mouseleave when the hovered element changes.
            // Per W3C spec, mouseenter/mouseleave do NOT bubble but must fire
            // independently on every ancestor between the old/new target and
            // their lowest common ancestor.
            if ( target != mHoverElement )
            {
                // Build ancestor chains for old and new hover elements.
                auto ancestors = [] ( Element * e ) -> std::vector<Element*>
                {
                    std::vector<Element*> chain;
                    for ( Node * n = e; n != nullptr; n = n->parentNode() )
                    {
                        if ( n->nodeType() == Node::ELEMENT_NODE )
                        {
                            chain.push_back ( static_cast<Element*> ( n ) );
                        }
                    }
                    return chain;
                };
                auto oldChain = ancestors ( mHoverElement );  // [target, parent, ... , root]
                auto newChain = ancestors ( target );

                // Find the lowest common ancestor (first element in oldChain also in newChain).
                Element* lca = nullptr;
                for ( auto * e : oldChain )
                {
                    if ( std::find ( newChain.begin(), newChain.end(), e ) != newChain.end() )
                    {
                        lca = e;
                        break;
                    }
                }

                // Fire mouseleave on old target and ancestors up to (not including) LCA.
                for ( auto * e : oldChain )
                {
                    if ( e == lca )
                    {
                        break;
                    }
                    e->setHover ( false );
                    e->ReselectCSS();
                    MouseEvent leaveEvent ( "mouseleave", MouseEventInit{EventModifierInit{UIEventInit{EventInit{false, false, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, 0, aButtons, target} );
                    e->dispatchEvent ( leaveEvent );
                }

                // Fire mouseenter on new target and ancestors up to (not including) LCA.
                // Fire in top-down order (ancestors first), per spec.
                std::vector<Element*> enterElements;
                for ( auto * e : newChain )
                {
                    if ( e == lca )
                    {
                        break;
                    }
                    enterElements.push_back ( e );
                }
                for ( auto it = enterElements.rbegin(); it != enterElements.rend(); ++it )
                {
                    ( *it )->setHover ( true );
                    ( *it )->ReselectCSS();
                    MouseEvent enterEvent ( "mouseenter", MouseEventInit{EventModifierInit{UIEventInit{EventInit{false, false, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, 0, aButtons, mHoverElement} );
                    ( *it )->dispatchEvent ( enterEvent );
                }

                mHoverElement = target;
            }
            if ( target )
            {
                MouseEvent moveEvent ( "mousemove", MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, 0, aButtons, nullptr} );
                target->dispatchEvent ( moveEvent );
            }
        }

        void Window::HandleMouseDown ( double aX, double aY, short aButton,
                                       unsigned short aButtons,
                                       bool aCtrlKey, bool aShiftKey,
                                       bool aAltKey, bool aMetaKey )
        {
            Element* target = mDocument.elementFromPoint ( mCanvas, aX, aY );
            // Update focus on mousedown
            if ( target != mFocusedElement )
            {
                if ( mFocusedElement )
                {
                    mFocusedElement->setFocus ( false );
                    mFocusedElement->ReselectCSS();
                    FocusEvent blurEvent ( "blur", FocusEventInit{UIEventInit{EventInit{false, false, false}, this, 0}, target} );
                    mFocusedElement->dispatchEvent ( blurEvent );
                    FocusEvent focusOutEvent ( "focusout", FocusEventInit{UIEventInit{EventInit{true, false, false}, this, 0}, target} );
                    mFocusedElement->dispatchEvent ( focusOutEvent );
                }
                mFocusedElement = target;
                if ( mFocusedElement )
                {
                    mFocusedElement->setFocus ( true );
                    mFocusedElement->ReselectCSS();
                    FocusEvent focusInEvent ( "focusin", FocusEventInit{UIEventInit{EventInit{true, false, false}, this, 0}, nullptr} );
                    mFocusedElement->dispatchEvent ( focusInEvent );
                    FocusEvent focusEvent ( "focus", FocusEventInit{UIEventInit{EventInit{false, false, false}, this, 0}, nullptr} );
                    mFocusedElement->dispatchEvent ( focusEvent );
                }
            }
            // Set :active state
            if ( mActiveElement && mActiveElement != target )
            {
                mActiveElement->setActive ( false );
                mActiveElement->ReselectCSS();
            }
            mActiveElement = target;
            if ( mActiveElement )
            {
                mActiveElement->setActive ( true );
                mActiveElement->ReselectCSS();
            }
            if ( target )
            {
                MouseEvent downEvent ( "mousedown", MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, aButton, aButtons, nullptr} );
                target->dispatchEvent ( downEvent );
            }
        }

        void Window::HandleMouseUp ( double aX, double aY, short aButton,
                                     unsigned short aButtons,
                                     bool aCtrlKey, bool aShiftKey,
                                     bool aAltKey, bool aMetaKey )
        {
            // Clear :active state
            if ( mActiveElement )
            {
                mActiveElement->setActive ( false );
                mActiveElement->ReselectCSS();
                mActiveElement = nullptr;
            }
            Element* target = mDocument.elementFromPoint ( mCanvas, aX, aY );
            if ( target )
            {
                MouseEvent upEvent ( "mouseup", MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, aButton, aButtons, nullptr} );
                target->dispatchEvent ( upEvent );
                // Fire click after mouseup on the same element
                MouseEvent clickEvent ( "click", MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 1}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, aButton, aButtons, nullptr} );
                target->dispatchEvent ( clickEvent );
            }
        }

        void Window::HandleKeyDown ( const DOMString& aKey, const DOMString& aCode,
                                     unsigned long aLocation, bool aRepeat,
                                     bool aCtrlKey, bool aShiftKey,
                                     bool aAltKey, bool aMetaKey )
        {
            EventTarget* target = mFocusedElement ? static_cast<EventTarget*> ( mFocusedElement ) : static_cast<EventTarget*> ( this );
            KeyboardEvent keyDownEvent ( "keydown", KeyboardEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aKey, aCode, aLocation, aRepeat, false} );
            target->dispatchEvent ( keyDownEvent );
        }

        void Window::HandleKeyUp ( const DOMString& aKey, const DOMString& aCode,
                                   unsigned long aLocation,
                                   bool aCtrlKey, bool aShiftKey,
                                   bool aAltKey, bool aMetaKey )
        {
            EventTarget* target = mFocusedElement ? static_cast<EventTarget*> ( mFocusedElement ) : static_cast<EventTarget*> ( this );
            KeyboardEvent keyUpEvent ( "keyup", KeyboardEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aKey, aCode, aLocation, false, false} );
            target->dispatchEvent ( keyUpEvent );
        }

        void Window::HandleWheel ( double aX, double aY,
                                   double aDeltaX, double aDeltaY,
                                   unsigned long aDeltaMode,
                                   unsigned short aButtons,
                                   bool aCtrlKey, bool aShiftKey,
                                   bool aAltKey, bool aMetaKey )
        {
            Element* target = mDocument.elementFromPoint ( mCanvas, aX, aY );
            if ( target )
            {
                WheelEvent wheelEvent ( "wheel", WheelEventInit{MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, 0, aButtons, nullptr}, aDeltaX, aDeltaY, 0.0, aDeltaMode} );
                target->dispatchEvent ( wheelEvent );
            }
        }
    }
}