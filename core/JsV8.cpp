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

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "aeongui/JsV8.h"
#include "aeongui/Window.h"
#include "aeongui/Document.h"

namespace AeonGUI
{
    void Console_Log ( const v8::FunctionCallbackInfo<v8::Value>& args )
    {
        v8::Isolate* isolate = args.GetIsolate();
        v8::HandleScope scope ( isolate );
        for ( int i = 0; i < args.Length(); ++i )
        {
            if ( i > 0 )
            {
                std::cout << " ";
            }
            v8::String::Utf8Value utf8 ( isolate, args[i] );
            std::cout << *utf8;
        }
        std::cout << std::endl;
        args.GetReturnValue().Set ( args.Holder() );
    }

    V8::V8 ( Window* aWindow, Document* aDocument )
    {
        // Create a new Isolate and make it the current one.
        v8::Isolate::CreateParams create_params;
        create_params.array_buffer_allocator = v8::ArrayBuffer::Allocator::NewDefaultAllocator();
        mIsolate = IsolatePtr{v8::Isolate::New ( create_params ) };
        {
            v8::Isolate::Scope isolate_scope ( mIsolate.get() );
            v8::HandleScope handle_scope ( mIsolate.get() );

            // Create Global Object Template
            v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New ( mIsolate.get() );
            global->SetInternalFieldCount ( 1 );

            // Create Console Object Template
            v8::Handle<v8::ObjectTemplate> console = v8::ObjectTemplate::New ( mIsolate.get() );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate.get(), "log" ), v8::FunctionTemplate::New ( mIsolate.get(), Console_Log ) );

            // Create Context
            v8::Local<v8::Context> context = v8::Context::New ( mIsolate.get(), nullptr, global );
            mGlobalContext.Reset ( mIsolate.get(), context );
            v8::Context::Scope context_scope ( context );

            // Store the Window pointer at the global object
            context->Global()->SetInternalField ( 0, v8::External::New ( mIsolate.get(), aWindow ) );

            // Proxy the global object thru the window property
            context->Global()->Set ( context,
                                     v8::String::NewFromUtf8Literal ( mIsolate.get(), "window" ),
                                     context->Global() ).Check();

            // Add the console object to the global object
            context->Global()->Set ( context,
                                     v8::String::NewFromUtf8Literal ( mIsolate.get(), "console" ),
                                     console->NewInstance ( context ).ToLocalChecked() ).Check();
        }
    }

    void V8::CreateObject ( Node* aNode )
    {
    }

    V8::~V8()
    {
        mGlobalContext.Reset();
    }

    void V8::Eval ( const std::string& aString )
    {
        v8::Isolate::Scope isolate_scope ( mIsolate.get() );
        v8::HandleScope handle_scope ( mIsolate.get() );
        v8::Local<v8::Context> context =
            v8::Local<v8::Context>::New ( mIsolate.get(), mGlobalContext );
        v8::Context::Scope context_scope ( context );
        v8::Local<v8::String> source =
            v8::String::NewFromUtf8 ( mIsolate.get(), aString.data(),
                                      v8::NewStringType::kNormal )
            .ToLocalChecked();
        v8::Local<v8::Script> script =
            v8::Script::Compile ( context, source ).ToLocalChecked();
#if 0
        v8::Local<v8::Value> result = script->Run ( context ).ToLocalChecked();
        v8::String::Utf8Value utf8 ( mIsolate.get(), result );
        std::cout << *utf8 << std::endl;
#else
        /**@todo Eval should return a value,
         * but it must an engine independent wrapper.*/
        script->Run ( context ).ToLocalChecked();
#endif
    }
}
