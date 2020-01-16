/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_JAVASCRIPT_H
#define AEONGUI_JAVASCRIPT_H
#include "aeongui/Platform.h"
#include <string>
namespace AeonGUI
{
    class JavaScript
    {
    public:
        virtual void Eval ( const std::string& aString ) = 0;
        DLL virtual ~JavaScript() = 0;
    };
}
#endif
