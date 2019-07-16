/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_ATTRIBUTEMAP_H
#define AEONGUI_ATTRIBUTEMAP_H
#include "aeongui/Color.h"
#include <string>
#include <unordered_map>
#include <variant>
namespace AeonGUI
{
    using AttributeType = std::variant<double, Color>;
    using AttributeMap = std::unordered_map<std::string, AttributeType>;
}
#endif
