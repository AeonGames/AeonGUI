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
#ifndef AEONGUI_WEBIDL_PARSER_INTERFACE_H
#define AEONGUI_WEBIDL_PARSER_INTERFACE_H

#include <string>
#include <vector>
#include "Attribute.h"
namespace AeonGUI
{
    class Interface
    {
    public:
        Interface ( const std::string& aName, const std::string& aInheritance, std::vector<Attribute> aAttributes ) :
            mName{aName}, mInheritance{aInheritance}, mAttributes{aAttributes} {}
        const std::string& GetName() const
        {
            return mName;
        }
        const std::string& GetInheritance() const
        {
            return mInheritance;
        }
        const std::vector<Attribute>& GetAttributes() const
        {
            return mAttributes;
        }
    private:
        std::string mName{};
        std::string mInheritance{};
        std::vector<Attribute> mAttributes{};
    };
}
#endif
