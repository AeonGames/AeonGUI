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
#ifndef AEONGUI_DOM_LOCATION_H
#define AEONGUI_DOM_LOCATION_H
#include "aeongui/dom/USVString.hpp"
#include "aeongui/Platform.hpp"
#include <vector>

namespace AeonGUI
{
    namespace DOM
    {
        class Location
        {
        public:
            DLL Location();
            DLL ~Location();
            // Methods to manipulate the URL
            DLL void assign ( const USVString& url );
            DLL void replace ( const USVString& url );
            void reload();
            // Attributes to access various parts of the URL
            DLL const USVString& href() const;
            DLL const USVString& origin() const;
            DLL const USVString& protocol() const;
            DLL const USVString& host() const;
            DLL const USVString& hostname() const;
            DLL const USVString& port() const;
            DLL const USVString& pathname() const;
            DLL const USVString& search() const;
            DLL const USVString& hash() const;
            // Ancestor origins for security context
            DLL const std::vector<USVString>& ancestorOrigins() const;
        };
    }
}
#endif