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
#ifndef AEONGUI_DOM_EVENTTARGET_H
#define AEONGUI_DOM_EVENTTARGET_H
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <functional>
#include <unordered_map>
#include "aeongui/Platform.hpp"
#include "AnyType.hpp"
#include "DOMString.hpp"
#include "EventListener.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class AbortSignal;
        class Event;
        class EventTarget;
        /** @brief Callback type for event handlers. */
        using EventHandler = std::function<void ( Event& ) >;
        /** @brief Options for addEventListener / removeEventListener. */
        struct EventListenerOptions
        {
            bool capture{false}; ///< If true, listen during the capture phase.
        };

        /** @brief Extended options for addEventListener. */
        struct AddEventListenerOptions : EventListenerOptions
        {
            bool passive;        ///< If true, the listener will not call preventDefault.
            bool once{false};    ///< If true, the listener is automatically removed after firing.
            AbortSignal* signal{nullptr}; ///< An AbortSignal that can remove the listener.
        };

        /** @brief Base class for objects that can receive DOM events.
         *
         *  Implements the DOM EventTarget interface: addEventListener,
         *  removeEventListener, and dispatchEvent.
         */
        class EventTarget
        {
        public:
            /** @brief Virtual destructor. */
            virtual ~EventTarget() = 0;
            /** @brief Register an event listener.
             *  @param type     The event type to listen for.
             *  @param callback The EventListener to invoke.
             *  @param options  Listener options (capture, once, etc.).
             */
            void addEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, AddEventListenerOptions, bool>& options = {} );
            /** @brief Unregister an event listener.
             *  @param type     The event type.
             *  @param callback The EventListener to remove.
             *  @param options  Matching options used during registration.
             */
            void removeEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, EventListenerOptions, bool>& options = {} );
            /** @brief Dispatch an event to this target.
             *  @param event The event to dispatch.
             *  @return true if the event was not cancelled.
             */
            virtual bool dispatchEvent ( Event& event );
        private:
            std::unordered_map<DOMString, std::vector<EventListener* >> mEventListeners{};
        };

        /** @brief Signal that can abort an asynchronous operation.
         *
         *  Implements the DOM AbortSignal interface. Can be used to
         *  remove event listeners by associating one with a signal.
         */
        class AbortSignal :
            EventTarget
        {
        public:
            /** @brief Create an already-aborted signal.
             *  @param reason Optional reason for the abort.
             *  @return An aborted AbortSignal.
             */
            static AbortSignal abort ( const std::optional<AnyType>& reason = std::nullopt );
            /** @brief Create a signal that will abort after a timeout.
             *  @param milliseconds Time in milliseconds.
             *  @return A timeout AbortSignal.
             */
            static AbortSignal timeout ( uint64_t milliseconds );
            /** @brief Create a signal that aborts when any of the given signals abort.
             *  @param signals The signals to monitor.
             *  @return A composite AbortSignal.
             */
            static AbortSignal any ( std::vector<AbortSignal> signals );

            /** @brief Check whether the signal has been aborted.
             *  @return true if aborted. */
            bool aborted() const;
            /** @brief Get the reason for the abort.
             *  @return The abort reason. */
            const AnyType & reason() const;
            /** @brief Throw if the signal has been aborted. */
            void throwIfAborted();

            EventHandler onabort; ///< Handler invoked when the signal is aborted.
            /** @brief Destructor. */
            virtual ~AbortSignal() = default;
        private:
            AbortSignal() = default;
            bool m_aborted;
            AnyType m_reason;
        };
    }
}
#endif
