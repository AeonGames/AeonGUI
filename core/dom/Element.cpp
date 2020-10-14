/*
Copyright (C) 2010-2013,2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include <string>
#include "Element.h"
#include "aeongui/Color.h"

namespace AeonGUI
{
    int ParseStyle ( AttributeMap& aAttributeMap, const char* s );
    Element::Element ( const std::string& aTagName, const AttributeMap& aAttributes ) : mTagName{aTagName}, mAttributeMap{aAttributes}
    {
        auto style = mAttributeMap.find ( "style" );
        if ( style != mAttributeMap.end() )
        {
            if ( ParseStyle ( mAttributeMap, std::get<std::string> ( style->second ).c_str() ) )
            {
                auto id = mAttributeMap.find ( "id" );
                if ( id != mAttributeMap.end() )
                {
                    std::cerr << "In Element id = " << std::get<std::string> ( id->second ) << std::endl;
                }
                std::cerr << "Error parsing style: " << std::get<std::string> ( style->second ) << std::endl;
            }
        }
    }

    Element::~Element() = default;

    AttributeType Element::GetAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        auto i = mAttributeMap.find ( attrName );
        if ( i != mAttributeMap.end() )
        {
            return i->second;
        }
        return aDefault;
    }

    void Element::SetAttribute ( const char* attrName, const AttributeType& aValue )
    {
        mAttributeMap[attrName] = aValue;
    }

    AttributeType Element::GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        AttributeType attr = GetAttribute ( attrName );
        Node* parent = parentNode();
        while ( std::holds_alternative<std::monostate> ( attr ) && parent != nullptr )
        {
            if ( parent->nodeType() == ELEMENT_NODE )
            {
                attr = reinterpret_cast<Element*> ( parent )->GetAttribute ( attrName );
            }
            parent = parent->parentNode();
        }
        return std::holds_alternative<std::monostate> ( attr ) ? aDefault : attr;
    }

    Node::NodeType Element::nodeType() const
    {
        return ELEMENT_NODE;
    }
    const std::string& Element::tagName() const
    {
        return mTagName;
    }

    void Element::Initialize ( v8::Isolate* aIsolate )
    {
        if ( HasFunctionTemplate ( aIsolate, typeid ( Element ) ) )
        {
            throw std::runtime_error ( "Isolate already initialized." );
        }

        v8::Local<v8::Context> context = aIsolate->GetCurrentContext();

        // Store constructor on a callback data object
        v8::Local<v8::ObjectTemplate> constructor_data_template = v8::ObjectTemplate::New ( aIsolate );
        constructor_data_template->SetInternalFieldCount ( 1 );
        v8::Local<v8::Object> constructor_data =
            constructor_data_template->NewInstance ( context ).ToLocalChecked();

        // Prepare Element constructor template
        v8::Local<v8::FunctionTemplate> constructor_template = v8::FunctionTemplate::New ( aIsolate, Element::New, constructor_data );
        constructor_template->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, "Element" ).ToLocalChecked() );
        constructor_template->Inherit ( GetFunctionTemplate ( aIsolate, typeid ( Node ) ) );
        constructor_template->InstanceTemplate()->SetInternalFieldCount ( 1 );
        constructor_template->InstanceTemplate()->SetHandler ( v8::NamedPropertyHandlerConfiguration{GetAttribute, SetAttribute} );

        AddFunctionTemplate ( aIsolate, typeid ( Element ), constructor_template );

        v8::Local<v8::Function> constructor = constructor_template->GetFunction ( context ).ToLocalChecked();
        constructor_data->SetInternalField ( 0, constructor );
        context->Global()->Set ( context, v8::String::NewFromUtf8 (
                                     aIsolate, "Element" ).ToLocalChecked(),
                                 constructor ).FromJust();
    }

    void Element::Finalize ( v8::Isolate* aIsolate )
    {
        RemoveFunctionTemplate ( aIsolate, typeid ( Element ) );
    }

    void Element::New ( const v8::FunctionCallbackInfo<v8::Value>& aArgs )
    {
        v8::Isolate* isolate = aArgs.GetIsolate();
        v8::Local<v8::Context> context = isolate->GetCurrentContext();

        if ( aArgs.IsConstructCall() )
        {
            // Invoked as constructor
            Element* obj = Construct ( *v8::String::Utf8Value{isolate, aArgs[0]}, {} );
            obj->Wrap ( aArgs.This() );
            aArgs.GetReturnValue().Set ( aArgs.This() );
        }
        else
        {
            std::vector<v8::Local<v8::Value>> args{};
            args.reserve ( aArgs.Length() );
            for ( auto i = 0; i < aArgs.Length(); ++i )
            {
                args.emplace_back ( aArgs[i] );
            }
            v8::Local<v8::Function> constructor =
                aArgs.Data().As<v8::Object>()->GetInternalField ( 0 ).As<v8::Function>();
            v8::Local<v8::Object> result =
                constructor->NewInstance ( context, static_cast<int> ( args.size() ), args.data() ).ToLocalChecked();
            aArgs.GetReturnValue().Set ( result );
        }
    }

    void Element::GetAttribute ( v8::Local<v8::Name> property, const v8::PropertyCallbackInfo<v8::Value>& info )
    {
        Element* element = JsObjectWrap::Unwrap<Element> ( info.Holder() );
        v8::String::Utf8Value key ( info.GetIsolate(), property );
        AttributeType attribute = element->GetAttribute ( *key );

        std::visit (
            [&info] ( auto&& arg )
        {
            using T = std::decay_t<decltype ( arg ) >;
            if constexpr ( std::is_same_v<T, std::string> )
            {
                info.GetReturnValue().Set ( v8::String::NewFromUtf8 ( info.GetIsolate(), arg.c_str() ).ToLocalChecked() );
            }
            else if constexpr ( std::is_same_v<T, double> )
            {
                info.GetReturnValue().Set ( v8::Number::New ( info.GetIsolate(), arg ) );
            }
            else if constexpr ( std::is_same_v<T, ColorAttr> )
            {
                if ( std::holds_alternative<std::monostate> ( arg ) )
                {
                    info.GetReturnValue().Set ( v8::String::NewFromUtf8Literal ( info.GetIsolate(), "none" ) );
                }
                else
                {
                    std::string string_color = std::get<Color> ( arg ).ToString();
                    info.GetReturnValue().Set ( v8::String::NewFromUtf8 ( info.GetIsolate(), string_color.c_str() ).ToLocalChecked() );
                }
            }
        }, attribute );
    }

    void Element::SetAttribute ( v8::Local<v8::Name> name, v8::Local<v8::Value> value,
                                 const v8::PropertyCallbackInfo<v8::Value>& info )
    {
        if ( name->IsSymbol() )
        {
            return;
        }
        Element* element = JsObjectWrap::Unwrap<Element> ( info.Holder() );
        std::string key{*v8::String::Utf8Value{info.GetIsolate(), name}};
        if ( value->IsString() )
        {
            uint32_t color{};
            if ( Color::IsColor ( *v8::String::Utf8Value{info.GetIsolate(), value}, &color ) )
            {
                element->mAttributeMap[key] = Color{color};
            }
            else
            {
                element->mAttributeMap[key] = *v8::String::Utf8Value{info.GetIsolate(), value};
            }
        }
        else if ( value->IsNumber() )
        {
            element->mAttributeMap[key] = v8::Number::Cast ( *value )->NumberValue ( info.GetIsolate()->GetCurrentContext() ).FromJust();
        }
        else
        {
            return;
        }
        info.GetReturnValue().Set ( value );
    }
}
