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
#ifndef AEONGUI_WEBIDL_PARSER_ATTRIBUTE_H
#define AEONGUI_WEBIDL_PARSER_ATTRIBUTE_H

#include <string>
namespace AeonGUI
{
    class Attribute
    {
    public:
        Attribute ( const std::string& aName, const std::string& aType, bool aReadOnly = false, bool aPossiblyNull = false ) :
            mName{aName}, mType{aType}, mReadOnly{aReadOnly}, mPossiblyNull{aPossiblyNull} {}
        const std::string& GetName() const
        {
            return mName;
        }
        const std::string& GetType() const
        {
            return mType;
        }
        bool IsReadOnly() const
        {
            return mReadOnly;
        }
        bool IsPossiblyNull() const
        {
            return mPossiblyNull;
        }
    private:
        std::string mName{};
        std::string mType{};
        bool mReadOnly{};
        bool mPossiblyNull{};
    };
}
#endif
