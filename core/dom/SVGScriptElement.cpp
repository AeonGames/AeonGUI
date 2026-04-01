/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#include <algorithm>
#include <filesystem>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef LoadLibrary
#else
#include <dlfcn.h>
#endif
#include "aeongui/dom/SVGScriptElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/dom/Text.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        // ---- CallbackEventListener: bridges a C callback to EventListener ----

        class CallbackEventListener : public EventListener
        {
        public:
            CallbackEventListener ( AeonGUI_EventCallback aCallback, void* aUserData )
                : mCallback ( aCallback ), mUserData ( aUserData ) {}
            ~CallbackEventListener() override = default;

            void handleEvent ( Event& event ) override
            {
                if ( mCallback )
                {
                    mCallback ( reinterpret_cast<AeonGUI_Event*> ( &event ), mUserData );
                }
            }

            AeonGUI_EventCallback mCallback{};
            void* mUserData{};
        };

        // ---- Platform library helpers ----

#ifdef _WIN32
        static constexpr const char* LibExtension = ".dll";
        static constexpr const char* LibPrefix = "";
#elif __APPLE__
        static constexpr const char* LibExtension = ".dylib";
        static constexpr const char* LibPrefix = "lib";
#else
        static constexpr const char* LibExtension = ".so";
        static constexpr const char* LibPrefix = "lib";
#endif

        static std::string FileURLToPath ( const std::string& url )
        {
            const std::string prefix = "file://";
            if ( url.compare ( 0, prefix.size(), prefix ) != 0 )
            {
                return url;
            }
            std::string path = url.substr ( prefix.size() );
#ifdef _WIN32
            // Strip leading slash before drive letter: /C:/path -> C:/path
            if ( path.size() >= 3 && path[0] == '/' && std::isalpha ( static_cast<unsigned char> ( path[1] ) ) && path[2] == ':' )
            {
                path = path.substr ( 1 );
            }
#else
            // Collapse extra leading slashes from file:////... into a normal absolute path.
            while ( path.size() > 1 && path[0] == '/' && path[1] == '/' )
            {
                path.erase ( path.begin() );
            }
#endif
            return path;
        }

        // ---- Thread-local active script element for static callbacks ----

        thread_local SVGScriptElement* SVGScriptElement::sActiveScriptElement = nullptr;

        // ---- Construction / Destruction ----

        SVGScriptElement::SVGScriptElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGElement ( aTagName, std::move ( aAttributes ), aParent )
        {
        }

        SVGScriptElement::~SVGScriptElement()
        {
            UnloadLibrary();
        }

        bool SVGScriptElement::IsDrawEnabled() const
        {
            return false;
        }

        // ---- Library path construction ----

        std::vector<std::string> SVGScriptElement::BuildLibraryPaths ( const std::string& aHref ) const
        {
            // Extract directory from the document URL and convert to a system path
            std::string directory;
            Document* doc = ownerDocument();
            if ( doc )
            {
                std::string path = FileURLToPath ( doc->url() );
                auto lastSep = path.find_last_of ( "/\\" );
                if ( lastSep != std::string::npos )
                {
                    directory = path.substr ( 0, lastSep + 1 );
                }
            }
            std::vector<std::string> candidates;
            candidates.push_back ( directory + LibPrefix + aHref + LibExtension );
#ifndef _WIN32
            if ( std::string ( LibPrefix ).size() )
            {
                candidates.push_back ( directory + aHref + LibExtension );
            }
#endif
            return candidates;
        }

        // ---- Shared library loading ----

        void SVGScriptElement::LoadLibrary ( const std::string& aPath )
        {
#ifdef _WIN32
            mLibHandle = static_cast<void*> ( ::LoadLibraryA ( aPath.c_str() ) );
            if ( !mLibHandle )
            {
                std::cerr << "SVGScriptElement: Failed to load library: " << aPath
                          << " (error " << GetLastError() << ")" << std::endl;
                return;
            }
            mOnLoadFunc = reinterpret_cast<AeonGUI_OnLoadFunc> (
                              ::GetProcAddress ( static_cast<HMODULE> ( mLibHandle ), "AeonGUI_OnLoad" ) );
            mOnUnloadFunc = reinterpret_cast<AeonGUI_OnUnloadFunc> (
                                ::GetProcAddress ( static_cast<HMODULE> ( mLibHandle ), "AeonGUI_OnUnload" ) );
#else
            mLibHandle = dlopen ( aPath.c_str(), RTLD_LAZY );
            if ( !mLibHandle )
            {
                std::cerr << "SVGScriptElement: Failed to load library: " << aPath
                          << " (" << dlerror() << ")" << std::endl;
                return;
            }
            mOnLoadFunc = reinterpret_cast<AeonGUI_OnLoadFunc> (
                              dlsym ( mLibHandle, "AeonGUI_OnLoad" ) );
            mOnUnloadFunc = reinterpret_cast<AeonGUI_OnUnloadFunc> (
                                dlsym ( mLibHandle, "AeonGUI_OnUnload" ) );
#endif
            if ( !mOnLoadFunc )
            {
                std::cerr << "SVGScriptElement: Library " << aPath
                          << " does not export AeonGUI_OnLoad" << std::endl;
                UnloadLibrary();
            }
        }

        void SVGScriptElement::UnloadLibrary()
        {
            // Remove all registered callback listeners
            for ( auto& reg : mCallbacks )
            {
                if ( reg.element && reg.listener )
                {
                    static_cast<EventTarget*> ( reg.element )->removeEventListener (
                        reg.eventType, reg.listener.get() );
                }
            }
            mCallbacks.clear();

            if ( mLibHandle )
            {
#ifdef _WIN32
                ::FreeLibrary ( static_cast<HMODULE> ( mLibHandle ) );
#else
                dlclose ( mLibHandle );
#endif
                mLibHandle = nullptr;
            }
            mOnLoadFunc = nullptr;
            mOnUnloadFunc = nullptr;
        }

        // ---- OnLoad / OnUnload lifecycle ----

        void SVGScriptElement::OnLoad()
        {
            auto typeIt = mAttributes.find ( "type" );
            if ( typeIt == mAttributes.end() || typeIt->second != "native" )
            {
                return;
            }

            auto hrefIt = mAttributes.find ( "href" );
            if ( hrefIt == mAttributes.end() || hrefIt->second.empty() )
            {
                std::cerr << "SVGScriptElement: native script missing href attribute" << std::endl;
                return;
            }

            auto candidates = BuildLibraryPaths ( hrefIt->second );
            std::string libPath = candidates.front();

            for ( const auto& candidate : candidates )
            {
                if ( std::filesystem::exists ( candidate ) )
                {
                    libPath = candidate;
                    break;
                }
            }

            std::cout << "SVGScriptElement: Loading native plugin: " << libPath << std::endl;
            LoadLibrary ( libPath );

            if ( !mLibHandle )
            {
                for ( const auto& candidate : candidates )
                {
                    if ( candidate == libPath )
                    {
                        continue;
                    }
                    std::cout << "SVGScriptElement: Retrying native plugin: " << candidate << std::endl;
                    LoadLibrary ( candidate );
                    if ( mLibHandle )
                    {
                        break;
                    }
                }
            }

            if ( mOnLoadFunc )
            {
                Document* doc = ownerDocument();
                mContext.document = reinterpret_cast<AeonGUI_Document*> ( doc );
                mContext.getElementById = &SVGScriptElement::API_getElementById;
                mContext.addEventListener = &SVGScriptElement::API_addEventListener;
                mContext.removeEventListener = &SVGScriptElement::API_removeEventListener;
                mContext.getAttribute = &SVGScriptElement::API_getAttribute;
                mContext.getEventType = &SVGScriptElement::API_getEventType;
                mContext.setAttribute = &SVGScriptElement::API_setAttribute;
                mContext.querySelector = &SVGScriptElement::API_querySelector;

                sActiveScriptElement = this;
                mOnLoadFunc ( &mContext );
                sActiveScriptElement = nullptr;
            }
        }

        void SVGScriptElement::OnUnload()
        {
            if ( mOnUnloadFunc )
            {
                sActiveScriptElement = this;
                mOnUnloadFunc ( &mContext );
                sActiveScriptElement = nullptr;
            }
            UnloadLibrary();
        }

        // ---- Static C API implementations ----

        AeonGUI_Element* SVGScriptElement::API_getElementById ( AeonGUI_Document* doc, const char* id )
        {
            if ( !doc || !id )
            {
                return nullptr;
            }
            Document* document = reinterpret_cast<Document*> ( doc );
            Element* element = document->getElementById ( id );
            return reinterpret_cast<AeonGUI_Element*> ( element );
        }

        void SVGScriptElement::API_addEventListener ( AeonGUI_Element* element, const char* type,
                AeonGUI_EventCallback callback, void* userData )
        {
            if ( !element || !type || !callback || !sActiveScriptElement )
            {
                return;
            }
            Element* elem = reinterpret_cast<Element*> ( element );

            auto listener = std::make_unique<CallbackEventListener> ( callback, userData );
            EventListener* raw = listener.get();
            sActiveScriptElement->mCallbacks.push_back ( { elem, type, std::move ( listener ) } );

            static_cast<EventTarget*> ( elem )->addEventListener ( type, raw );
        }

        void SVGScriptElement::API_removeEventListener ( AeonGUI_Element* element, const char* type,
                AeonGUI_EventCallback callback, void* userData )
        {
            if ( !element || !type || !callback || !sActiveScriptElement )
            {
                return;
            }
            Element* elem = reinterpret_cast<Element*> ( element );

            auto& callbacks = sActiveScriptElement->mCallbacks;
            auto it = std::find_if ( callbacks.begin(), callbacks.end(),
                                     [elem, type, callback, userData] ( const RegisteredCallback & reg )
            {
                return reg.element == elem &&
                       reg.eventType == type &&
                       reg.listener->mCallback == callback &&
                       reg.listener->mUserData == userData;
            } );

            if ( it != callbacks.end() )
            {
                static_cast<EventTarget*> ( elem )->removeEventListener ( type, it->listener.get() );
                callbacks.erase ( it );
            }
        }

        const char* SVGScriptElement::API_getAttribute ( AeonGUI_Element* element, const char* name )
        {
            if ( !element || !name )
            {
                return nullptr;
            }
            Element* elem = reinterpret_cast<Element*> ( element );
            const DOMString* value = elem->getAttribute ( name );
            return value ? value->c_str() : nullptr;
        }

        void SVGScriptElement::API_setAttribute ( AeonGUI_Element* element, const char* name, const char* value )
        {
            if ( !element || !name || !value )
            {
                return;
            }
            Element* elem = reinterpret_cast<Element*> ( element );
            if ( std::string ( name ) == "textContent" )
            {
                // textContent is a DOM property, not an attribute.
                // Replace all child text nodes with a single new one.
                while ( !elem->childNodes().empty() )
                {
                    elem->RemoveNode ( elem->childNodes().front().get() );
                }
                elem->AddNode ( std::make_unique<Text> ( value, elem ) );
                Document* doc = elem->ownerDocument();
                if ( doc )
                {
                    doc->MarkDirty();
                }
            }
            else
            {
                elem->setAttribute ( name, value );
            }
        }

        const char* SVGScriptElement::API_getEventType ( AeonGUI_Event* event )
        {
            if ( !event )
            {
                return nullptr;
            }
            Event* evt = reinterpret_cast<Event*> ( event );
            return evt->type().c_str();
        }

        AeonGUI_Element* SVGScriptElement::API_querySelector ( AeonGUI_Element* element, const char* selector )
        {
            if ( !element || !selector )
            {
                return nullptr;
            }
            Element* elem = reinterpret_cast<Element*> ( element );
            Element* result = elem->querySelector ( selector );
            return reinterpret_cast<AeonGUI_Element*> ( result );
        }
    }
}
