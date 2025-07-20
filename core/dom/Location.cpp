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
#include "aeongui/dom/Location.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        Location::Location() = default;
        Location::~Location() = default;

        void Location::assign ( const USVString& url )
        {
            // Implementation for assigning a new URL
        }

        void Location::replace ( const USVString& url )
        {
            // Implementation for replacing the current URL
        }

        void Location::reload()
        {
            // Implementation for reloading the current URL
        }

        const USVString& Location::href() const
        {
            // Return the full URL as a string
            static USVString href;
            return href;
        }

        const USVString& Location::origin() const
        {
            // Return the origin of the URL
            static USVString origin;
            return origin;
        }

        const USVString& Location::protocol() const
        {
            // Return the protocol part of the URL
            static USVString protocol;
            return protocol;
        }

        const USVString& Location::host() const
        {
            // Return the host part of the URL
            static USVString host;
            return host;
        }

        const USVString& Location::hostname() const
        {
            // Return the hostname part of the URL
            static USVString hostname;
            return hostname;
        }

        const USVString& Location::port() const
        {
            // Return the port part of the URL
            static USVString port;
            return port;
        }

        const USVString& Location::pathname() const
        {
            // Return the pathname part of the URL
            static USVString pathname;
            return pathname;
        }

        const USVString& Location::search() const
        {
            // Return the search/query part of the URL
            static USVString search;
            return search;
        }

        const USVString& Location::hash() const
        {
            // Return the hash fragment of the URL
            static USVString hash;
            return hash;
        }

        const std::vector<DOMString>& Location::ancestorOrigins() const
        {
            // Return a list of ancestor origins for security context, if applicable
            static std::vector<DOMString> ancestorOrigins;
            return ancestorOrigins;
        }
    }
}