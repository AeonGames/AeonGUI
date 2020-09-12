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
#ifndef AEONGUI_V8_H
#define AEONGUI_V8_H
#include "aeongui/Platform.h"
#include "aeongui/JavaScript.h"
#include "v8-platform.h"
#include "v8.h"

namespace AeonGUI
{
    class Node;
    class Window;
    class Document;

    struct IsolateDeleter
    {
        void operator() ( std::unique_ptr<v8::Isolate>::pointer p )
        {
            p->Dispose();
        }
    };
    using IsolatePtr =  std::unique_ptr<v8::Isolate, IsolateDeleter>;

    class V8 : public JavaScript
    {
    public:
        V8 ( Window* aWindow, Document* aDocument );
        ~V8() final;
        void Eval ( const std::string& aString ) final;
        void CreateObject ( Node* aNode );
    private:
        IsolatePtr mIsolate{};
        v8::Persistent<v8::Context> mGlobalContext{};
    };
}
#endif
