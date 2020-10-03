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
#include "SVGElement.h"

namespace AeonGUI
{
    namespace DOM
    {
        SVGElement::SVGElement ( const std::string& aTagName, const AttributeMap& aAttributes ) : Element { aTagName, aAttributes } {}
        SVGElement::~SVGElement() = default;
        void SVGElement::Initialize ( v8::Isolate* aIsolate )
        {
            if ( AeonGUI::JsObjectWrap::HasFunctionTemplate ( aIsolate, typeid ( SVGElement ) ) )
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
            constructor_template->SetClassName ( v8::String::NewFromUtf8 ( aIsolate, "SVGElement" ).ToLocalChecked() );
            constructor_template->InstanceTemplate()->SetInternalFieldCount ( 1 );
            constructor_template->Inherit ( GetFunctionTemplate ( aIsolate, typeid ( Element ) ) );

            AddFunctionTemplate ( aIsolate, typeid ( SVGElement ), constructor_template );

            v8::Local<v8::Function> constructor = constructor_template->GetFunction ( context ).ToLocalChecked();
            constructor_data->SetInternalField ( 0, constructor );
            context->Global()->Set ( context, v8::String::NewFromUtf8 (
                                         aIsolate, "SVGElement" ).ToLocalChecked(),
                                     constructor ).FromJust();
        }

        void SVGElement::Finalize ( v8::Isolate* aIsolate )
        {
            RemoveFunctionTemplate ( aIsolate, typeid ( SVGElement ) );
        }
    }
}
