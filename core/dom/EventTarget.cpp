/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include "EventTarget.h"
#include "Event.h"
namespace AeonGUI
{
    namespace DOM
    {
        void EventTarget::addEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, AddEventListenerOptions, bool>& options )
        {
            if ( mEventListeners.find ( type ) == mEventListeners.end() )
            {
                mEventListeners[type] = std::vector<EventListener*>();
            }
            mEventListeners[type].push_back ( callback );
        }
        void EventTarget::removeEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, EventListenerOptions, bool>& options )
        {
            auto it = mEventListeners.find ( type );
            if ( it != mEventListeners.end() )
            {
                auto& listeners = it->second;
                listeners.erase ( std::remove ( listeners.begin(), listeners.end(), callback ), listeners.end() );
                if ( listeners.empty() )
                {
                    mEventListeners.erase ( it );
                }
            }
        }
        bool EventTarget::dispatchEvent ( Event& event )
        {
            auto it = mEventListeners.find ( event.type() );
            if ( it != mEventListeners.end() )
            {
                for ( auto& listener : it->second )
                {
                    listener->handleEvent ( event );
                }
                return true;
            }
            return false;
        }
    }
}
