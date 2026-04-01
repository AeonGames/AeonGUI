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
#ifdef AEONGUI_USE_SKIA
#include "aeongui/SkiaCanvas.hpp"
#else
#include "aeongui/CairoCanvas.hpp"
#endif
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/SVGGeometryElement.hpp"
#include "aeongui/dom/SVGFilterElement.hpp"
#include "aeongui/dom/SVGFEDropShadowElement.hpp"
#include "aeongui/dom/MouseEvent.hpp"
#include "aeongui/dom/KeyboardEvent.hpp"
#include "aeongui/dom/WheelEvent.hpp"
#include "aeongui/dom/FocusEvent.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace AeonGUI
{
    namespace DOM
    {
        Window::Window () : mCanvas{std::make_unique <
#ifdef AEONGUI_USE_SKIA
                                        SkiaCanvas
#else
                                        CairoCanvas
#endif
                                        > () } {}
        Window::Window ( uint32_t aWidth, uint32_t aHeight ) :
            mCanvas{std::make_unique <
#ifdef AEONGUI_USE_SKIA
                    SkiaCanvas
#else
                    CairoCanvas
#endif
                    > ( aWidth, aHeight ) }
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
            mCanvas->ResizeViewport ( aWidth, aHeight );
            mDocument.MarkDirty();
        }

        const uint8_t* Window::GetPixels() const
        {
            return mCanvas->GetPixels();
        }

        size_t Window::GetWidth() const
        {
            return mCanvas->GetWidth();
        }
        size_t Window::GetHeight() const
        {
            return mCanvas->GetHeight();
        }
        size_t Window::GetStride() const
        {
            return mCanvas->GetStride();
        }

        bool Window::Draw()
        {
            if ( !mDocument.IsDirty() )
            {
                return false;
            }
            if ( mDocument.IsFullDirty() )
            {
                FullDraw();
            }
            else
            {
                PartialDraw();
            }
            mDocument.ClearDirty();
            return true;
        }

        void Window::AssignPickIds()
        {
            mPickIdCounter = 0;
            mPickElements.fill ( nullptr );
            mDocument.Draw ( *mCanvas, [this] ( const Node & aNode )
            {
                if ( aNode.nodeType() == Node::ELEMENT_NODE &&
                     dynamic_cast<const SVGGeometryElement * > ( &aNode ) &&
                     mPickIdCounter < 255 )
                {
                    ++mPickIdCounter;
                    mPickElements[mPickIdCounter] = const_cast<Element*> ( static_cast<const Element*> ( &aNode ) );
                    mCanvas->SetPickId ( mPickIdCounter );
                }
                else
                {
                    mCanvas->SetPickId ( 0 );
                }
            } );
            mCanvas->SetPickId ( 0 );
        }

        void Window::CacheBounds()
        {
            mCachedBounds.clear();
            for ( uint8_t i = 1; i <= mPickIdCounter; ++i )
            {
                if ( mPickElements[i] )
                {
                    Canvas::PickBounds bounds = mCanvas->GetPickBounds ( i );
                    // Expand bounds for drop-shadow filter effects.
                    const DOMString* filterAttr = mPickElements[i]->getAttribute ( "filter" );
                    if ( filterAttr && filterAttr->compare ( 0, 5, "url(#" ) == 0 && filterAttr->back() == ')' )
                    {
                        std::string filterId = filterAttr->substr ( 5, filterAttr->size() - 6 );
                        Element* filterElem = mDocument.getElementById ( filterId );
                        if ( filterElem && filterElem->tagName() == "filter" )
                        {
                            for ( const auto& child : filterElem->childNodes() )
                            {
                                if ( child->nodeType() == Node::ELEMENT_NODE &&
                                     static_cast<Element * > ( child.get() )->tagName() == "feDropShadow" )
                                {
                                    const auto* ds = static_cast<const SVGFEDropShadowElement*> ( child.get() );
                                    double expandX = std::abs ( ds->dx() ) + 3.0 * ds->stdDeviationX();
                                    double expandY = std::abs ( ds->dy() ) + 3.0 * ds->stdDeviationY();
                                    bounds.x1 -= expandX;
                                    bounds.y1 -= expandY;
                                    bounds.x2 += expandX;
                                    bounds.y2 += expandY;
                                    break;
                                }
                            }
                        }
                    }
                    mCachedBounds[mPickElements[i]] = bounds;
                }
            }
        }

        void Window::FullDraw()
        {
            mCanvas->Clear();
            mCanvas->ResetPick();
            AssignPickIds();
            CacheBounds();
        }

        void Window::PartialDraw()
        {
            // Compute the union of dirty element AABBs in device space.
            double dx1 = 1e30, dy1 = 1e30, dx2 = -1e30, dy2 = -1e30;
            bool hasBounds = false;
            size_t dirtyCount = mDocument.GetDirtyElements().size();
            size_t foundCount = 0;
            for ( const Element * e : mDocument.GetDirtyElements() )
            {
                auto it = mCachedBounds.find ( e );
                if ( it != mCachedBounds.end() )
                {
                    dx1 = std::min ( dx1, it->second.x1 );
                    dy1 = std::min ( dy1, it->second.y1 );
                    dx2 = std::max ( dx2, it->second.x2 );
                    dy2 = std::max ( dy2, it->second.y2 );
                    hasBounds = true;
                    ++foundCount;
                }
            }
            if ( !hasBounds || dirtyCount > foundCount )
            {
                // Some dirty elements lack cached bounds (e.g. text nodes)
                // — fall back to full draw.
                mCanvas->Clear();
                mCanvas->ResetPick();
                AssignPickIds();
                CacheBounds();
                return;
            }
            // Clamp to viewport.
            double vw = static_cast<double> ( mCanvas->GetWidth() );
            double vh = static_cast<double> ( mCanvas->GetHeight() );
            dx1 = std::max ( std::floor ( dx1 ), 0.0 );
            dy1 = std::max ( std::floor ( dy1 ), 0.0 );
            dx2 = std::min ( std::ceil ( dx2 ), vw );
            dy2 = std::min ( std::ceil ( dy2 ), vh );
            if ( dx1 >= dx2 || dy1 >= dy2 )
            {
                // Degenerate dirty rect — nothing to redraw.
                return;
            }
            // Clip both surfaces to the dirty rectangle, clear, and redraw.
            mCanvas->Save();
            mCanvas->SetClipRect ( dx1, dy1, dx2 - dx1, dy2 - dy1 );
            mCanvas->Clear();
            mCanvas->ResetPick();
            AssignPickIds();
            mCanvas->Restore();
            CacheBounds();
        }

        void Window::Update ( double aDeltaTime )
        {
            mDocument.AdvanceTime ( aDeltaTime );
        }

        Element* Window::elementFromPoint ( double aX, double aY ) const
        {
            uint8_t id = mCanvas->PickAtPoint ( aX, aY );
            return ( id > 0 && id <= mPickIdCounter ) ? mPickElements[id] : nullptr;
        }

        void Window::HandleMouseMove ( double aX, double aY, unsigned short aButtons,
                                       bool aCtrlKey, bool aShiftKey,
                                       bool aAltKey, bool aMetaKey )
        {
            Element* target = elementFromPoint ( aX, aY );
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
            Element* target = elementFromPoint ( aX, aY );
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
            Element* target = elementFromPoint ( aX, aY );
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
            Element* target = elementFromPoint ( aX, aY );
            if ( target )
            {
                WheelEvent wheelEvent ( "wheel", WheelEventInit{MouseEventInit{EventModifierInit{UIEventInit{EventInit{true, true, false}, this, 0}, aCtrlKey, aShiftKey, aAltKey, aMetaKey}, aX, aY, aX, aY, 0, aButtons, nullptr}, aDeltaX, aDeltaY, 0.0, aDeltaMode} );
                target->dispatchEvent ( wheelEvent );
            }
        }
    }
}