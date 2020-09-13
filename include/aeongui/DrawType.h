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
#ifndef AEONGUI_DRAWTYPE_H
#define AEONGUI_DRAWTYPE_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <initializer_list>
#include <variant>
#include "aeongui/Platform.h"
#include "aeongui/Vector2.h"

namespace AeonGUI
{
    using DrawType = std::variant<uint64_t, double, bool>;
}
#endif
