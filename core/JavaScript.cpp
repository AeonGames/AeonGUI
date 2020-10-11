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
#include "aeongui/AeonGUI.h"
#include "aeongui/JavaScript.h"
#include "aeongui/Window.h"
#include "aeongui/Document.h"
#include "aeongui/ElementFactory.h"
#include "dom/EventTarget.h"
#include "dom/Node.h"
#include "dom/Element.h"
#include "libplatform/libplatform.h"

namespace AeonGUI
{
    static std::unique_ptr<v8::Platform> gPlatform{};
    bool InitializeJavaScript ( int argc, char *argv[] )
    {
        /** @todo Provide a plugin loading mechanism,
         *  so different JavaScript engines can be used */
        // Initialize V8.
        v8::V8::InitializeICU();
        v8::V8::InitializeExternalStartupData ( argv[0] );
        gPlatform = v8::platform::NewDefaultPlatform();
        v8::V8::InitializePlatform ( gPlatform.get() );
        v8::V8::Initialize();
        return true;
    }

    void FinalizeJavaScript()
    {
        v8::V8::Dispose();
        v8::V8::ShutdownPlatform();
        gPlatform.reset();
    }

    static void log ( const v8::FunctionCallbackInfo<v8::Value>& info )
    {
        v8::Isolate* isolate = info.GetIsolate();
        v8::HandleScope scope ( isolate );
        for ( int i = 0; i < info.Length(); ++i )
        {
            if ( i > 0 )
            {
                std::cout << " ";
            }
            v8::String::Utf8Value utf8 ( isolate, info[i] );
            std::cout << *utf8;
        }
        std::cout << std::endl;
        info.GetReturnValue().Set ( info.Holder() );
    }

    static void createElementNS ( const v8::FunctionCallbackInfo<v8::Value>& info )
    {
        v8::Isolate* isolate = info.GetIsolate();
        v8::HandleScope scope ( isolate );
        info.GetReturnValue().Set ( info.Holder() );
    }

    static void getElementById ( const v8::FunctionCallbackInfo<v8::Value>& info )
    {
        v8::Isolate* isolate = info.GetIsolate();
        v8::HandleScope scope ( isolate );
        info.GetReturnValue().Set ( info.Holder() );
    }

    JavaScript::JavaScript ( Window* aWindow, Document* aDocument )
    {
        // Create a new Isolate and make it the current one.
        v8::Isolate::CreateParams create_params{};
        create_params.array_buffer_allocator = v8::ArrayBuffer::Allocator::NewDefaultAllocator();
        mIsolate = v8::Isolate::New ( create_params );
        {
            v8::Isolate::Scope isolate_scope ( mIsolate );
            v8::HandleScope handle_scope ( mIsolate );

            // Create Global Object Template
            v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New ( mIsolate );
            global->SetInternalFieldCount ( 1 );

            // Create Console Object Template
            v8::Handle<v8::ObjectTemplate> console = v8::ObjectTemplate::New ( mIsolate );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "log" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "warn" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "info" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "error" ), v8::FunctionTemplate::New ( mIsolate, log ) );

            // Create Document Object Template
            v8::Handle<v8::ObjectTemplate> document = v8::ObjectTemplate::New ( mIsolate );
            document->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "createElementNS" ), v8::FunctionTemplate::New ( mIsolate, createElementNS ) );
            document->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "getElementById" ), v8::FunctionTemplate::New ( mIsolate, getElementById ) );

            // Create Context
            v8::Local<v8::Context> context = v8::Context::New ( mIsolate, nullptr, global );
            mContext.Reset ( mIsolate, context );
            v8::Context::Scope context_scope ( context );

            // Store the Window pointer at the global object
            context->Global()->SetInternalField ( 0, v8::External::New ( mIsolate, aWindow ) );

            // Proxy the global object thru the window property
            context->Global()->Set ( context,
                                     v8::String::NewFromUtf8Literal ( mIsolate, "window" ),
                                     context->Global() ).Check();

            // Add the console object to the global object
            context->Global()->Set ( context,
                                     v8::String::NewFromUtf8Literal ( mIsolate, "console" ),
                                     console->NewInstance ( context ).ToLocalChecked() ).Check();

            // Add the document object to the global object
            context->Global()->Set ( context,
                                     v8::String::NewFromUtf8Literal ( mIsolate, "document" ),
                                     document->NewInstance ( context ).ToLocalChecked() ).Check();
            Initialize ( mIsolate );
        }
    }

    JavaScript::~JavaScript()
    {
        Finalize ( mIsolate );
        mContext.Reset();
        if ( mIsolate )
        {
            mIsolate->Dispose();
            mIsolate = nullptr;
        }
    }

    void JavaScript::Eval ( const std::string& aString )
    {
        v8::Isolate::Scope isolate_scope ( mIsolate );
        v8::HandleScope handle_scope ( mIsolate );
        v8::Local<v8::Context> context =
            v8::Local<v8::Context>::New ( mIsolate, mContext );
        v8::Context::Scope context_scope ( context );
        v8::Local<v8::String> source =
            v8::String::NewFromUtf8 ( mIsolate, aString.data(),
                                      v8::NewStringType::kNormal )
            .ToLocalChecked();
        v8::Local<v8::Script> script =
            v8::Script::Compile ( context, source ).ToLocalChecked();
#if 0
        v8::Local<v8::Value> result = script->Run ( context ).ToLocalChecked();
        v8::String::Utf8Value utf8 ( mIsolate, result );
        std::cout << *utf8 << std::endl;
#else
        /**@todo Eval should return a value,
         * but it must an engine independent wrapper.*/
        script->Run ( context ).ToLocalChecked();
#endif
    }
}
