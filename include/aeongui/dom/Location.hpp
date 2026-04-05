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
#ifndef AEONGUI_DOM_LOCATION_H
#define AEONGUI_DOM_LOCATION_H
#include "aeongui/dom/USVString.hpp"
#include "aeongui/Platform.hpp"
#include <vector>
#include <functional>
namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Represents the URL of the active document.
         *
         *  Provides access to the individual components of a URL (protocol,
         *  hostname, port, pathname, etc.) and methods to navigate.
         *  @see https://html.spec.whatwg.org/multipage/nav-history-apis.html#location
         */
        class Location
        {
        public:
            /** @brief Default constructor. URL is "about:blank". */
            AEONGUI_DLL Location();
            /** @brief Construct with a change callback.
             *  @param callback Called whenever the location changes.
             */
            AEONGUI_DLL Location ( std::function<void ( const Location& ) > callback );
            /** @brief Destructor. */
            AEONGUI_DLL ~Location();
            /**DOM Properties and Methods @{*/
            /** @brief Navigate to the given URL.
             *  @param url The URL to navigate to.
             */
            AEONGUI_DLL void assign ( const USVString& url );
            /** @brief Replace the current URL without creating a history entry.
             *  @param url The URL to navigate to.
             */
            AEONGUI_DLL void replace ( const USVString& url );
            /** @brief Assign a URL using the = operator.
             *  @param url The URL to navigate to.
             *  @return Reference to this Location.
             */
            AEONGUI_DLL Location& operator= ( const USVString& url );
            /** @brief Reload the current document. */
            AEONGUI_DLL void reload();
            /** @brief Get the full URL.
             *  @return The href string. */
            AEONGUI_DLL const USVString& href() const;
            /** @brief Get the origin portion of the URL.
             *  @return The origin string. */
            AEONGUI_DLL const USVString& origin() const;
            /** @brief Get the protocol (e.g. "https:").
             *  @return The protocol string. */
            AEONGUI_DLL const USVString& protocol() const;
            /** @brief Get the host (hostname:port).
             *  @return The host string. */
            AEONGUI_DLL const USVString& host() const;
            /** @brief Get the hostname.
             *  @return The hostname string. */
            AEONGUI_DLL const USVString& hostname() const;
            /** @brief Get the port number.
             *  @return The port string. */
            AEONGUI_DLL const USVString& port() const;
            /** @brief Get the pathname.
             *  @return The pathname string. */
            AEONGUI_DLL const USVString& pathname() const;
            /** @brief Get the query string (including leading '?').
             *  @return The search string. */
            AEONGUI_DLL const USVString& search() const;
            /** @brief Get the fragment (including leading '#').
             *  @return The hash string. */
            AEONGUI_DLL const USVString& hash() const;
            /**@}*/
            /** @brief Set or replace the change callback.
             *  @param callback The new callback function.
             */
            AEONGUI_DLL void SetCallback ( std::function<void ( const Location& ) > callback );
        private:
            USVString m_href{"about:blank"};
            USVString m_origin{"about:blank"};
            USVString m_protocol{"about:"};
            USVString m_host{"blank"};
            USVString m_hostname{"blank"};
            USVString m_port{};
            USVString m_pathname{};
            USVString m_search{};
            USVString m_hash{};
            std::function<void ( const Location& ) > mCallback{};
        };
    }
}
#endif