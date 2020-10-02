/*
Copyright (C) 2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_EVENTTARGET_H
#define AEONGUI_EVENTTARGET_H
#include <unordered_map>
#include <vector>
#include <string>
#include "aeongui/Platform.h"
#include "aeongui/JsObjectWrap.h"

namespace AeonGUI
{
    class EventTarget : public JsObjectWrap
    {
    public:
        DLL void addEventListener ( const std::string& aType, v8::Local<v8::Function> aCallback );
        DLL void removeEventListener ( const std::string& aType, v8::Local<v8::Function> aCallback );
        DLL bool dispatchEvent ( v8::Local<v8::Object> aEvent );
        static DLL v8::Persistent<v8::FunctionTemplate, v8::CopyablePersistentTraits<v8::FunctionTemplate>>& GetFunctionTemplate ( v8::Isolate* aIsolate );
        static DLL void Initialize ( v8::Isolate* aIsolate );
        static DLL void Finalize ( v8::Isolate* aIsolate );
        static DLL void New ( const v8::FunctionCallbackInfo<v8::Value>& aArgs );
        static DLL void addEventListener ( const v8::FunctionCallbackInfo<v8::Value>& aArgs );
        static DLL void removeEventListener ( const v8::FunctionCallbackInfo<v8::Value>& aArgs );
        static DLL void dispatchEvent ( const v8::FunctionCallbackInfo<v8::Value>& aArgs );
    private:
        std::unordered_map <
        std::string,
            std::vector<v8::Persistent<v8::Function, v8::CopyablePersistentTraits<v8::Function>>>
            > mEventListeners{};
    };
}
#endif
