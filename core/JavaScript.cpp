/*
Copyright (C) 2019,2021 Rodrigo Jose Hernandez Cordoba

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
#include <array>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "aeongui/AeonGUI.h"
#include "aeongui/JavaScript.h"
#include "aeongui/JsObjectWrap.h"
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

    enum GlobalInternalFields
    {
        WINDOW = 0,
        LOCATION,
        COUNT
    };

    static const std::regex number{ "-?([0-9]+|[0-9]*\\.[0-9]+([eE][-+]?[0-9]+)?)" };
    static AttributeMap ExtractElementAttributes ( xmlElementPtr aXmlElementPtr )
    {
        AttributeMap attribute_map{};
        for ( xmlNodePtr attribute = reinterpret_cast<xmlNodePtr> ( aXmlElementPtr->attributes ); attribute; attribute = attribute->next )
        {
            std::cmatch match;
            const char* value = reinterpret_cast<const char*> ( xmlGetProp ( reinterpret_cast<xmlNodePtr> ( aXmlElementPtr ), attribute->name ) );
            if ( std::regex_match ( value, match, number ) )
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = std::stod ( match[0].str() );
            }
            else if ( std::regex_match ( value, match, Color::ColorRegex ) )
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = Color{ match[0].str() };
            }
            else
            {
                attribute_map[reinterpret_cast<const char*> ( attribute->name )] = value;
            }
        }
        return attribute_map;
    }

    static void AddNodes ( Node* aNode, xmlNode* aXmlNode )
    {
        for ( xmlNode* node = aXmlNode; node; node = node->next )
        {
            if ( node->type == XML_ELEMENT_NODE )
            {
                xmlElementPtr element = reinterpret_cast<xmlElementPtr> ( node );
                AddNodes ( aNode->AddNode ( Construct ( reinterpret_cast<const char*> ( element->name ), ExtractElementAttributes ( element ) ) ), node->children );
            }
#if 0
            else if ( xmlNodeIsText ( node ) && !xmlIsBlankNode ( node ) )
            {
                AddNodes ( aNode->AddNode ( new Text{ reinterpret_cast<const char*> ( node->content ) } ), node->children );
            }
#endif
        }
    }

    bool InitializeJavaScript ( int argc, char *argv[] )
    {
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

    /** @note document.location and window.location *SHOULD* be a Location object
     * according to the W3C specification but since we're not interested in dealing with remote
     * resources yet, we'll deal with it as a simple string for now.
     */
    static void GetLocation ( v8::Local<v8::String> property,
                              const v8::PropertyCallbackInfo<v8::Value>& info )
    {
        v8::Isolate* isolate = info.GetIsolate();
        info.GetReturnValue().Set ( isolate->GetCurrentContext()->Global()->GetInternalField ( GlobalInternalFields::LOCATION ) );
    }

    static void SetLocation ( v8::Local<v8::String> property, v8::Local<v8::Value> value,
                              const v8::PropertyCallbackInfo<void>& info )
    {
        v8::Isolate* isolate = info.GetIsolate();
        v8::String::Utf8Value utf8 ( isolate, value );
        std::cout << __FUNCTION__ << "( \"" << *utf8 << "\" )" << std::endl;
        v8::Local<v8::Context> context = isolate->GetCurrentContext();
        v8::Local<v8::Object> global = context->Global();
        global->SetInternalField ( GlobalInternalFields::LOCATION, value );
        v8::Local<v8::Object> document = global->Get ( context, v8::String::NewFromUtf8Literal ( isolate, "document" ) ).ToLocalChecked().As<v8::Object>();
        xmlDocPtr xml{ xmlReadFile ( *utf8, nullptr, 0 ) };

        if ( xml == nullptr )
        {
            /** @todo This should be a Js exception. */
            throw std::runtime_error ( "Could not parse xml file" );
        }

        xmlElementPtr root_element = reinterpret_cast<xmlElementPtr> ( xmlDocGetRootElement ( xml ) );
        std::array<v8::Local<v8::Value>, 2> args
        {
            v8::String::NewFromUtf8Literal ( isolate, "http://www.w3.org/2000/svg" ),
            v8::String::NewFromUtf8 ( isolate, reinterpret_cast<const char*> ( root_element->name ) ).ToLocalChecked(),
        };

        v8::Local<v8::Object> documentElement = document->Get ( context, v8::String::NewFromUtf8Literal ( isolate, "createElementNS" ) ).ToLocalChecked().As<v8::Function>()->Call ( context, document, args.size(), args.data() ).ToLocalChecked().As<v8::Object>();
#if 0
        mDocumentElement = Construct ( reinterpret_cast<const char*> ( root_element->name ), ExtractElementAttributes ( root_element ) );
        AddNodes ( mDocumentElement, root_element->children );
        xmlFreeDoc ( document );
#endif
        document->Set ( context, v8::String::NewFromUtf8Literal ( isolate, "documentElement" ), documentElement ).Check();

        /**@todo Emit onload event.*/
    }

    JavaScript::JavaScript ( Window* aWindow )
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
            global->SetInternalFieldCount ( GlobalInternalFields::COUNT );

            // Create Console Object Template
            v8::Handle<v8::ObjectTemplate> console = v8::ObjectTemplate::New ( mIsolate );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "log" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "warn" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "info" ), v8::FunctionTemplate::New ( mIsolate, log ) );
            console->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "error" ), v8::FunctionTemplate::New ( mIsolate, log ) );

            // Create Document Object Template
            v8::Handle<v8::ObjectTemplate> document = v8::ObjectTemplate::New ( mIsolate );
            document->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "createElementNS" ), v8::FunctionTemplate::New ( mIsolate, AeonGUI::createElementNS ) );
            document->Set ( v8::String::NewFromUtf8Literal ( mIsolate, "getElementById" ), v8::FunctionTemplate::New ( mIsolate, getElementById ) );
            document->SetAccessor ( v8::String::NewFromUtf8Literal ( mIsolate, "location" ), AeonGUI::GetLocation, AeonGUI::SetLocation );
            global->SetAccessor ( v8::String::NewFromUtf8Literal ( mIsolate, "location" ), AeonGUI::GetLocation, AeonGUI::SetLocation );

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
         * but it must be an engine independent wrapper.*/
        script->Run ( context ).ToLocalChecked();
#endif
    }

    void JavaScript::SetLocation ( const std::string& aString )
    {
        v8::Isolate::Scope isolate_scope ( mIsolate );
        v8::HandleScope handle_scope ( mIsolate );
        v8::Local<v8::Context> context =
            v8::Local<v8::Context>::New ( mIsolate, mContext );
        v8::Context::Scope context_scope ( context );
        context->Global()->Set ( context,
                                 v8::String::NewFromUtf8Literal ( mIsolate, "location" ),
                                 v8::String::NewFromUtf8 ( mIsolate, aString.data(), v8::NewStringType::kNormal ).ToLocalChecked() ).Check();
    }

    Element* JavaScript::GetDocumentElement()
    {
#if 0
        /*
        This function was added so Window can retrieve a pointer to the document element
        and trigger drawing of the SVG document, but it should really be private and
        have a separate draw function.
        */
        v8::Isolate::Scope isolate_scope ( mIsolate );
        v8::HandleScope handle_scope ( mIsolate );
        v8::Local<v8::Context> context =
            v8::Local<v8::Context>::New ( mIsolate, mContext );
        v8::Context::Scope context_scope ( context );
        v8::Local<v8::Value> document_element = context->Global()->Get ( context, v8::String::NewFromUtf8Literal ( mIsolate, "document" ) ).ToLocalChecked();
        // Need to get document.documentElement here.
        if ( document_element->IsObject() && document_element->IsExternal() )
        {
            return JsObjectWrap::Unwrap<Element> ( document_element.As<v8::Object>() );
        }
#endif
        return nullptr;
    }
}
