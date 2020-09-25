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
#include "aeongui/JsObjectWrap.h"

namespace AeonGUI
{
    static void WeakCallback (
        const v8::WeakCallbackInfo<JsObjectWrap>& aInfo )
    {
        JsObjectWrap* wrap = aInfo.GetParameter();
        assert ( wrap->GetReferenceCount() == 0 );
        wrap->GetPersistentHandle().Reset();
        delete wrap;
    }

    JsObjectWrap::~JsObjectWrap()
    {
        if ( GetPersistentHandle().IsEmpty() )
        {
            return;
        }
        GetPersistentHandle().ClearWeak();
        GetPersistentHandle().Reset();
    }

    uint32_t JsObjectWrap::GetReferenceCount() const
    {
        return mReferenceCount;
    }

    JsObjectWrap* JsObjectWrap::Unwrap ( v8::Handle<v8::Object> handle )
    {
        assert ( !handle.IsEmpty() );
        assert ( handle->InternalFieldCount() > 0 );
        void* ptr = handle->GetAlignedPointerFromInternalField ( 0 );
        return static_cast<JsObjectWrap*> ( ptr );
    }

    v8::Local<v8::Object> JsObjectWrap::GetHandle() const
    {
        return GetHandle ( v8::Isolate::GetCurrent() );
    }

    v8::Local<v8::Object> JsObjectWrap::GetHandle ( v8::Isolate* isolate ) const
    {
        return v8::Local<v8::Object>::New ( isolate, GetPersistentHandle() );
    }

    v8::Persistent<v8::Object>& JsObjectWrap::GetPersistentHandle() const
    {
        return const_cast<v8::Persistent<v8::Object>&> ( mHandle );
    }

    void JsObjectWrap::Wrap ( v8::Handle<v8::Object> aHandle )
    {
        assert ( GetPersistentHandle().IsEmpty() );
        assert ( aHandle->InternalFieldCount() > 0 );
        aHandle->SetAlignedPointerInInternalField ( 0, this );
        GetPersistentHandle().Reset ( v8::Isolate::GetCurrent(), aHandle );
        MakeWeak();
    }

    void JsObjectWrap::MakeWeak()
    {
        GetPersistentHandle().SetWeak ( this, WeakCallback, v8::WeakCallbackType::kParameter );
    }

    void JsObjectWrap::Ref()
    {
        assert ( !GetPersistentHandle().IsEmpty() );
        GetPersistentHandle().ClearWeak();
        mReferenceCount++;
    }

    void JsObjectWrap::Unref()
    {
        assert ( !GetPersistentHandle().IsEmpty() );
        assert ( !GetPersistentHandle().IsWeak() );
        assert ( mReferenceCount > 0 );
        if ( mReferenceCount > 0 && --mReferenceCount == 0 )
        {
            MakeWeak();
        }
    }
}
