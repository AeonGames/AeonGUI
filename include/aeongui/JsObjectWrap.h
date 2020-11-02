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
#ifndef AEONGUI_JSOBJECTWRAP_H
#define AEONGUI_JSOBJECTWRAP_H

#include <v8.h>
#include <cassert>
#include <typeinfo>
#include "aeongui/Platform.h"
#include "aeongui/ElementFactory.h"

namespace AeonGUI
{
    /**
     * @brief JavaScript C++ Object Wrapper
     * based on https://github.com/nodejs/node/blob/master/src/node_object_wrap.h
     * @todo All uses of v8 classes and objects should be wrapped in a generic interface if JavaScript engines are to be exchangeable.
     */
    class JsObjectWrap
    {
    public:
        DLL virtual ~JsObjectWrap() = 0;

        DLL v8::Local<v8::Object> GetHandle() const;

        DLL v8::Local<v8::Object> GetHandle ( v8::Isolate* isolate ) const;

        DLL v8::Persistent<v8::Object>& GetPersistentHandle() const;

        DLL static JsObjectWrap* Unwrap ( v8::Handle<v8::Object> handle );
        template <class T>
        static T* Unwrap ( v8::Handle<v8::Object> aHandle )
        {
            return static_cast<T*> ( Unwrap ( aHandle ) );
        }
        DLL uint32_t GetReferenceCount() const;

        template<class T>
        static void New ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
        {
            v8::Isolate* isolate = aArgs.GetIsolate();
            v8::Local<v8::Context> context = isolate->GetCurrentContext();

            if ( aArgs.IsConstructCall() )
            {
                // Invoked as constructor
                T* obj = new T{};
                obj->Wrap ( aArgs.This() );
                aArgs.GetReturnValue().Set ( aArgs.This() );
            }
            else
            {
                // Invoked as plain function, turn into construct call.
                const int argc = 1;
                v8::Local<v8::Value> argv[argc] = { aArgs[0] };
                v8::Local<v8::Function> constructor =
                    aArgs.Data().As<v8::Object>()->GetInternalField ( 0 ).As<v8::Function>();
                v8::Local<v8::Object> result =
                    constructor->NewInstance ( context, argc, argv ).ToLocalChecked();
                aArgs.GetReturnValue().Set ( result );
            }
        }

        template<class T, class Parent = T>
        static void Initialize ( v8::Isolate* aIsolate, const std::string& aClassName )
        {
            if ( HasFunctionTemplate ( aIsolate, typeid ( T ) ) )
            {
                throw std::runtime_error ( "Isolate already initialized." );
            }

            v8::Local<v8::Context> context = aIsolate->GetCurrentContext();

            // Create constructor template
            v8::Local<v8::FunctionTemplate> constructor_template = v8::FunctionTemplate::New ( aIsolate );
            constructor_template->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, aClassName.c_str() ).ToLocalChecked() );
            if ( !std::is_same<T, Parent>::value )
            {
                constructor_template->Inherit ( GetFunctionTemplate ( aIsolate, typeid ( Parent ) ) );
            }
            AddFunctionTemplate ( aIsolate, typeid ( T ), constructor_template );
        }

        template<class T>
        static void Finalize ( v8::Isolate* aIsolate )
        {
            RemoveFunctionTemplate ( aIsolate, typeid ( T ) );
        }

    protected:
        DLL void Wrap ( v8::Handle<v8::Object> handle );
        DLL void MakeWeak();

        /** Marks the object as being attached to an event loop.
        * Refed objects will not be garbage collected, even if
        * all references are lost.
        */
        DLL void Ref();

        /** Marks an object as detached from the event loop.  This is its
        * default state.  When an object with a "weak" reference changes from
        * attached to detached state it will be freed. Be careful not to access
        * the object after making this call as it might be gone!
        * (A "weak reference" means an object that only has a
        * GetPersistentHandle handle.)
        *
        * DO NOT CALL THIS FROM DESTRUCTOR
        */
        DLL void Unref();
        DLL static v8::Local<v8::FunctionTemplate> GetFunctionTemplate ( v8::Isolate* aIsolate, const std::type_info& aTypeId );
        DLL static v8::Local<v8::FunctionTemplate> GetFunctionTemplateIfExists ( v8::Isolate* aIsolate, const std::type_info& aTypeId );
        DLL static bool HasFunctionTemplate ( v8::Isolate* aIsolate, const std::type_info& aTypeId );
        DLL static void AddFunctionTemplate ( v8::Isolate* aIsolate, const std::type_info& aTypeId, const v8::Local<v8::FunctionTemplate>& aFunctionTemplate );
        DLL static void RemoveFunctionTemplate ( v8::Isolate* aIsolate, const std::type_info& aTypeId );
    private:
        uint32_t mReferenceCount{};
        v8::Persistent<v8::Object> mHandle{};
    };
}
#endif
