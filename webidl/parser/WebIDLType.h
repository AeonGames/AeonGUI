/*
Copyright (C) 2021 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_WEBIDL_PARSER_TYPE_H
#define AEONGUI_WEBIDL_PARSER_TYPE_H

#include <string>
#include <vector>
#include <variant>
#include "Attribute.h"
#include "Interface.h"
namespace AeonGUI
{
    using WebIDLAtom = std::variant <
                       std::string,
                       Attribute,
                       Interface
                       >;
    using WebIDLType = std::variant<std::vector<WebIDLAtom>, WebIDLAtom>;
#define WEBIDLSTYPE AeonGUI::WebIDLType
}
#endif
