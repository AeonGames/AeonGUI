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
#include <iostream>
#include <unordered_map>
#include "EventTarget.h"
#include "aeongui/JsV8.h"

namespace AeonGUI
{
    static std::unordered_map<v8::Isolate*, v8::Persistent<v8::FunctionTemplate, v8::CopyablePersistentTraits<v8::FunctionTemplate>>> ConstructorsPerIsolate{};

    v8::Persistent<v8::FunctionTemplate, v8::CopyablePersistentTraits<v8::FunctionTemplate>>& EventTarget::GetFunctionTemplate ( v8::Isolate* aIsolate )
    {
        return ConstructorsPerIsolate.at ( aIsolate );
    }

    void EventTarget::InitIsolate ( v8::Isolate* aIsolate )
    {
        // Prepare constructor template
        v8::Local<v8::FunctionTemplate> tpl = v8::FunctionTemplate::New ( aIsolate, New );
        tpl->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, "EventTarget" ).ToLocalChecked() );
        tpl->InstanceTemplate()->SetInternalFieldCount ( 1 );

        tpl->PrototypeTemplate()->Set (
            aIsolate, "addEventListener",
            v8::FunctionTemplate::New ( aIsolate, addEventListener )
        );

        tpl->PrototypeTemplate()->Set (
            aIsolate, "removeEventListener",
            v8::FunctionTemplate::New ( aIsolate, removeEventListener )
        );

        tpl->PrototypeTemplate()->Set (
            aIsolate, "dispatchEvent",
            v8::FunctionTemplate::New ( aIsolate, dispatchEvent )
        );
        ConstructorsPerIsolate.emplace ( aIsolate, v8::Persistent<v8::FunctionTemplate> {aIsolate, tpl} );
    }

    void EventTarget::Finalize ( v8::Isolate* aIsolate )
    {
        ConstructorsPerIsolate.at ( aIsolate ).Reset();
        ConstructorsPerIsolate.erase ( aIsolate );
    }

    void EventTarget::InitContext ( v8::Local<v8::Context>& aContext )
    {
        v8::Isolate* isolate = aContext->GetIsolate();

        v8::Local<v8::FunctionTemplate> constructor_template =
            v8::Local<v8::FunctionTemplate>::New ( isolate, ConstructorsPerIsolate.at ( isolate ) );

        v8::Local<v8::Function> constructor = constructor_template->GetFunction ( aContext ).ToLocalChecked();
        aContext->Global()->Set ( aContext, v8::String::NewFromUtf8 (
                                      isolate, "EventTarget" ).ToLocalChecked(),
                                  constructor ).FromJust();
    }

    void EventTarget::New ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {
        v8::Isolate* isolate = aArgs.GetIsolate();
        v8::Local<v8::Context> context = isolate->GetCurrentContext();

        if ( aArgs.IsConstructCall() )
        {
            // Invoked as constructor
            EventTarget* obj = new EventTarget{};
            obj->Wrap ( aArgs.This() );
            aArgs.GetReturnValue().Set ( aArgs.This() );
        }
        else
        {
            // Invoked as plain function, turn into construct call.
            const int argc = 1;
            v8::Local<v8::Value> argv[argc] = { aArgs[0] };
            v8::Local<v8::Value> value = context->Global()->Get ( context, v8::String::NewFromUtf8 ( isolate, "EventTarget" ).ToLocalChecked() ).ToLocalChecked();
            v8::Local<v8::Function> cons = v8::Local<v8::Function>::Cast ( value );
            v8::Local<v8::Object> result =
                cons->NewInstance ( context, argc, argv ).ToLocalChecked();
            aArgs.GetReturnValue().Set ( result );
        }
    }

    void EventTarget::addEventListener ( const std::string& aType, v8::Local<v8::Function> aCallback )
    {
        auto it = mEventListeners.find ( aType );
        if ( it == mEventListeners.end() )
        {
            mEventListeners.emplace (
                aType,
                std::vector <
                v8::Persistent <
                v8::Function, v8::CopyablePersistentTraits <
                v8::Function
                >
                >
                >
            {{aCallback->GetIsolate(), aCallback}} );
        }
        else
        {
            /** @todo The event listener is appended to target’s event listener list
             * and is not appended if it has the same type, callback, and capture.
             * https://dom.spec.whatwg.org/#interface-eventtarget */
            ( *it ).second.emplace_back ( v8::Persistent<v8::Function> {aCallback->GetIsolate(), aCallback} );
        }
    }

    void EventTarget::removeEventListener ( const std::string& aType, v8::Local<v8::Function> aCallback )
    {
        auto it = mEventListeners.find ( aType );
        if ( it == mEventListeners.end() )
        {
            return;
        }
        if ( ( *it ).second.size() == 1 )
        {
            mEventListeners.erase ( it );
        }
        else
        {
            ( *it ).second.erase ( std::remove ( ( *it ).second.begin(), ( *it ).second.end(), aCallback ) );
        }
    }

    bool EventTarget::dispatchEvent ( v8::Local<v8::Object> aEvent )
    {
        return false;
    }

    void EventTarget::addEventListener ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {
        v8::Isolate* isolate = aArgs.GetIsolate();
        if ( aArgs.Length() != 2 || !aArgs[0]->IsString() || !aArgs[1]->IsFunction() )
        {
            isolate->ThrowException (
                v8::String::NewFromUtf8Literal ( aArgs.GetIsolate(), "addEventListener: Expected (string,function) as arguments" ) );
            return;
        }
        EventTarget* event_target = JsObjectWrap::Unwrap<EventTarget> ( aArgs.Holder() );
        event_target->addEventListener ( {*v8::String::Utf8Value{isolate, aArgs[0]}}, v8::Local<v8::Function>::Cast ( aArgs[1] ) );
        aArgs.GetReturnValue().Set ( v8::Undefined ( isolate ) );
    }
    void EventTarget::removeEventListener ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {
        v8::Isolate* isolate = aArgs.GetIsolate();
        if ( aArgs.Length() != 2 || !aArgs[0]->IsString() || !aArgs[1]->IsFunction() )
        {
            isolate->ThrowException (
                v8::String::NewFromUtf8Literal ( aArgs.GetIsolate(), "removeEventListener: Expected (string,function) as arguments" ) );
            return;
        }
        EventTarget* event_target = JsObjectWrap::Unwrap<EventTarget> ( aArgs.Holder() );
        event_target->removeEventListener ( {*v8::String::Utf8Value{isolate, aArgs[0]}}, v8::Local<v8::Function>::Cast ( aArgs[1] ) );
        aArgs.GetReturnValue().Set ( v8::Undefined ( isolate ) );
    }
    void EventTarget::dispatchEvent ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {}
}
