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
#include "aeongui/JavaScript.h"

namespace AeonGUI
{
    void EventTarget::Initialize ( v8::Isolate* aIsolate )
    {
        if ( HasFunctionTemplate ( aIsolate, typeid ( EventTarget ) ) )
        {
            throw std::runtime_error ( "Isolate already initialized." );
        }

        v8::Local<v8::Context> context = aIsolate->GetCurrentContext();

        // Prepare Event constructor template
        v8::Local<v8::FunctionTemplate> event_template = v8::FunctionTemplate::New ( aIsolate );
        event_template->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, "Event" ).ToLocalChecked() );
        /**@todo Add all official Event properties */
        event_template->PrototypeTemplate()->Set ( aIsolate, "type", v8::String::NewFromUtf8 ( aIsolate, "" ).ToLocalChecked() );

        //---------------------------------------
        // Store constructor on a callback data object
        v8::Local<v8::ObjectTemplate> constructor_data_template = v8::ObjectTemplate::New ( aIsolate );
        constructor_data_template->SetInternalFieldCount ( 1 );
        v8::Local<v8::Object> constructor_data =
            constructor_data_template->NewInstance ( context ).ToLocalChecked();

        // Prepare EventTarget constructor template
        v8::Local<v8::FunctionTemplate> constructor_template = v8::FunctionTemplate::New ( aIsolate, JsObjectWrap::New<EventTarget>, constructor_data );
        constructor_template->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, "EventTarget" ).ToLocalChecked() );
        constructor_template->InstanceTemplate()->SetInternalFieldCount ( 1 );

        constructor_template->PrototypeTemplate()->Set (
            aIsolate, "addEventListener",
            v8::FunctionTemplate::New ( aIsolate, addEventListener )
        );

        constructor_template->PrototypeTemplate()->Set (
            aIsolate, "removeEventListener",
            v8::FunctionTemplate::New ( aIsolate, removeEventListener )
        );

        constructor_template->PrototypeTemplate()->Set (
            aIsolate, "dispatchEvent",
            v8::FunctionTemplate::New ( aIsolate, dispatchEvent )
        );
        AddFunctionTemplate ( aIsolate, typeid ( EventTarget ), constructor_template );

        //----------------------------------------------------------------

        v8::Local<v8::Function> event = event_template->GetFunction ( context ).ToLocalChecked();
        context->Global()->Set ( context, v8::String::NewFromUtf8 (
                                     aIsolate, "Event" ).ToLocalChecked(),
                                 event ).FromJust();

        v8::Local<v8::Function> constructor = constructor_template->GetFunction ( context ).ToLocalChecked();
        constructor_data->SetInternalField ( 0, constructor );
        context->Global()->Set ( context, v8::String::NewFromUtf8 (
                                     aIsolate, "EventTarget" ).ToLocalChecked(),
                                 constructor ).FromJust();
    }

    void EventTarget::Finalize ( v8::Isolate* aIsolate )
    {
        RemoveFunctionTemplate ( aIsolate, typeid ( EventTarget ) );
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
        v8::Isolate* isolate = aEvent->GetIsolate();
        v8::Local<v8::Context> context = isolate->GetCurrentContext();

        auto it = mEventListeners.find ( {*v8::String::Utf8Value{isolate, aEvent->Get ( context, v8::String::NewFromUtf8 ( isolate, "type" ).ToLocalChecked() ).ToLocalChecked() }} );
        if ( it != mEventListeners.end() )
        {
            const int argc = 1;
            v8::Local<v8::Value> argv[argc] = { aEvent };
            for ( auto& i : it->second )
            {
                v8::Local<v8::Function> callback = v8::Local<v8::Function>::New ( isolate, i );
                callback->Call ( context, context->Global(), argc, argv ).ToLocalChecked();
            }
        }
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
    {
        v8::Isolate* isolate = aArgs.GetIsolate();
        if ( aArgs.Length() != 1 || !aArgs[0]->IsObject() )
        {
            isolate->ThrowException (
                v8::String::NewFromUtf8Literal ( aArgs.GetIsolate(), "removeEventListener: Expected (object) as arguments" ) );
            return;
        }
        v8::Local<v8::Object> event = v8::Local<v8::Object>::Cast ( aArgs[0] );
        EventTarget* event_target = JsObjectWrap::Unwrap<EventTarget> ( aArgs.Holder() );
        aArgs.GetReturnValue().Set ( v8::Boolean::New ( isolate, event_target->dispatchEvent ( event ) ) );
    }

    // JavaScript factory
    EventTarget* EventTarget::New ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {
        return new EventTarget;
    }
}
