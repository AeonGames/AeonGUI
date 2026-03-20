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
            DLL Location();
            /** @brief Construct with a change callback.
             *  @param callback Called whenever the location changes.
             */
            DLL Location ( std::function<void ( const Location& ) > callback );
            /** @brief Destructor. */
            DLL ~Location();
            /**DOM Properties and Methods @{*/
            /** @brief Navigate to the given URL.
             *  @param url The URL to navigate to.
             */
            DLL void assign ( const USVString& url );
            /** @brief Replace the current URL without creating a history entry.
             *  @param url The URL to navigate to.
             */
            DLL void replace ( const USVString& url );
            /** @brief Assign a URL using the = operator.
             *  @param url The URL to navigate to.
             *  @return Reference to this Location.
             */
            DLL Location& operator= ( const USVString& url );
            /** @brief Reload the current document. */
            DLL void reload();
            /** @brief Get the full URL. */
            DLL const USVString& href() const;
            /** @brief Get the origin portion of the URL. */
            DLL const USVString& origin() const;
            /** @brief Get the protocol (e.g. "https:"). */
            DLL const USVString& protocol() const;
            /** @brief Get the host (hostname:port). */
            DLL const USVString& host() const;
            /** @brief Get the hostname. */
            DLL const USVString& hostname() const;
            /** @brief Get the port number. */
            DLL const USVString& port() const;
            /** @brief Get the pathname. */
            DLL const USVString& pathname() const;
            /** @brief Get the query string (including leading '?'). */
            DLL const USVString& search() const;
            /** @brief Get the fragment (including leading '#'). */
            DLL const USVString& hash() const;
            /**@}*/
            /** @brief Set or replace the change callback.
             *  @param callback The new callback function.
             */
            DLL void SetCallback ( std::function<void ( const Location& ) > callback );
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