/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/dom/EventTarget.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/dom/Node.hpp"
#include <algorithm>
namespace AeonGUI
{
    namespace DOM
    {
        EventTarget::~EventTarget() = default;
        void EventTarget::addEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, AddEventListenerOptions, bool>& options )
        {
            bool capture = false;
            bool once = false;
            bool passive = false;
            if ( std::holds_alternative<bool> ( options ) )
            {
                capture = std::get<bool> ( options );
            }
            else if ( std::holds_alternative<AddEventListenerOptions> ( options ) )
            {
                const auto& opts = std::get<AddEventListenerOptions> ( options );
                capture = opts.capture;
                once = opts.once;
                passive = opts.passive;
            }
            auto& listeners = mEventListeners[type];
            // Don't add duplicate (same callback + capture)
            for ( const auto& entry : listeners )
            {
                if ( entry.callback == callback && entry.capture == capture )
                {
                    return;
                }
            }
            listeners.push_back ( {callback, capture, once, passive} );
        }
        void EventTarget::removeEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, EventListenerOptions, bool>& options )
        {
            bool capture = false;
            if ( std::holds_alternative<bool> ( options ) )
            {
                capture = std::get<bool> ( options );
            }
            else if ( std::holds_alternative<EventListenerOptions> ( options ) )
            {
                capture = std::get<EventListenerOptions> ( options ).capture;
            }
            auto it = mEventListeners.find ( type );
            if ( it != mEventListeners.end() )
            {
                auto& listeners = it->second;
                listeners.erase (
                    std::remove_if ( listeners.begin(), listeners.end(),
                                     [callback, capture] ( const RegisteredListener & entry )
                {
                    return entry.callback == callback && entry.capture == capture;
                } ),
                listeners.end() );
                if ( listeners.empty() )
                {
                    mEventListeners.erase ( it );
                }
            }
        }
        void EventTarget::invokeListeners ( Event& event, uint16_t phase )
        {
            auto it = mEventListeners.find ( event.type() );
            if ( it == mEventListeners.end() )
            {
                return;
            }
            event.m_currentTarget = this;
            // Iterate by index so additions/removals during dispatch don't invalidate iterators
            for ( auto& entry : it->second )
            {
                if ( event.m_stopImmediatePropagation )
                {
                    break;
                }
                // During capture phase, only invoke capture listeners.
                // During bubble phase, only invoke non-capture listeners.
                // At target, invoke all listeners.
                if ( phase == event.AT_TARGET ||
                     ( phase == event.CAPTURING_PHASE && entry.capture ) ||
                     ( phase == event.BUBBLING_PHASE && !entry.capture ) )
                {
                    entry.callback->handleEvent ( event );
                    if ( entry.once )
                    {
                        removeEventListener ( event.type(), entry.callback,
                                              EventListenerOptions{entry.capture} );
                    }
                }
            }
        }
        bool EventTarget::dispatchEvent ( Event& event )
        {
            // Build the propagation path: target -> ... -> root
            event.m_target = this;
            event.m_eventPhase = event.NONE;
            event.m_stopPropagation = false;
            event.m_stopImmediatePropagation = false;
            event.m_defaultPrevented = false;

            // Build path from target up to root by walking Node::parentNode()
            event.m_composedPath.clear();
            event.m_composedPath.push_back ( this );
            if ( auto * node = dynamic_cast<Node * > ( this ) )
            {
                for ( auto * parent = node->parentNode(); parent != nullptr; parent = parent->parentNode() )
                {
                    event.m_composedPath.push_back ( parent );
                }
            }

            // Capture phase: root to target (path is target..root, so iterate in reverse)
            event.m_eventPhase = event.CAPTURING_PHASE;
            for ( auto it = event.m_composedPath.rbegin(); it != event.m_composedPath.rend(); ++it )
            {
                if ( event.m_stopPropagation )
                {
                    break;
                }
                if ( *it == this )
                {
                    // At-target phase
                    event.m_eventPhase = event.AT_TARGET;
                }
                ( *it )->invokeListeners ( event, event.m_eventPhase );
            }

            // Bubble phase: target parent to root (skip target itself)
            if ( event.bubbles() && !event.m_stopPropagation && event.m_composedPath.size() > 1 )
            {
                event.m_eventPhase = event.BUBBLING_PHASE;
                // m_composedPath[0] is the target, [1..n] are ancestors
                for ( size_t i = 1; i < event.m_composedPath.size(); ++i )
                {
                    if ( event.m_stopPropagation )
                    {
                        break;
                    }
                    event.m_composedPath[i]->invokeListeners ( event, event.m_eventPhase );
                }
            }

            event.m_eventPhase = event.NONE;
            event.m_currentTarget = nullptr;
            return !event.defaultPrevented();
        }
    }
}
