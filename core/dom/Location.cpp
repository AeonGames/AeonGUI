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
#include <regex>
namespace AeonGUI
{
    namespace DOM
    {
        static const std::regex url_regex (
            R"(^((?:[/\w]+):)(//)?([^:/?#]+)?(?::(\d+))?([/\w :.-]+)?(\?[\w=&-]+)?(#[\w-]+)?$)"
        );
        Location::Location() = default;
        Location::~Location() = default;
        Location::Location ( std::function<void ( const Location& ) > callback )
            : mCallback ( std::move ( callback ) )
        {
        }

        void Location::SetCallback ( std::function<void ( const Location& ) > callback )
        {
            mCallback = std::move ( callback );
        }

        Location& Location::operator= ( const USVString& url )
        {
            assign ( url );
            return *this;
        }

        void Location::assign ( const USVString& url )
        {
            // Validate the URL format using regex
            std::smatch matches;
            if ( !std::regex_match ( url, matches, url_regex ) )
            {
                throw std::invalid_argument ( "Invalid URL format:" + url );
            }
            m_href = matches[0].str();
            m_protocol = matches[1].str();
            m_hostname = matches[3].str();
            m_port = matches[4].str();
            m_pathname = matches[5].str();
            m_search = matches[6].str();
            m_hash = matches[7].str();
            m_host = m_hostname + ( m_port.empty() ? "" : ":" + m_port );
            m_origin = m_protocol + matches[2].str() + m_host;
            if ( mCallback )
            {
                mCallback ( *this );
            }
        }

        void Location::replace ( const USVString& url )
        {
            // Implementation for replacing the current URL
            // We don't currently have a history, we may never support this
            // so replace just calls assign.
            assign ( url );
        }

        void Location::reload()
        {
            // Implementation for reloading the current URL
            if ( mCallback )
            {
                mCallback ( *this );
            }
        }

        const USVString& Location::href() const
        {
            // Return the full URL as a string
            return m_href;
        }

        const USVString& Location::origin() const
        {
            // Return the origin of the URL
            return m_origin;
        }

        const USVString& Location::protocol() const
        {
            // Return the protocol part of the URL
            return m_protocol;
        }

        const USVString& Location::host() const
        {
            // Return the host part of the URL
            return m_host;
        }

        const USVString& Location::hostname() const
        {
            // Return the hostname part of the URL
            return m_hostname;
        }

        const USVString& Location::port() const
        {
            // Return the port part of the URL
            return m_port;
        }

        const USVString& Location::pathname() const
        {
            // Return the pathname part of the URL
            return m_pathname;
        }

        const USVString& Location::search() const
        {
            // Return the search/query part of the URL
            return m_search;
        }

        const USVString& Location::hash() const
        {
            // Return the hash fragment of the URL
            return m_hash;
        }
    }
}