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
/**
 * @file SVGScriptElement.hpp
 * @brief SVG \<script\> element with native shared-library plugin support.
 */
#ifndef AEONGUI_SVGSCRIPTELEMENT_H
#define AEONGUI_SVGSCRIPTELEMENT_H

#include <string>
#include <array>
#include <vector>
#include <memory>
#include "SVGElement.hpp"
#include "aeongui/PluginAPI.h"

namespace AeonGUI
{
    namespace DOM
    {
        class CallbackEventListener;

        /** @brief SVG \<script\> element supporting native shared-library plugins.
         *
         *  When @c type is @c "native", loads a platform-specific shared library
         *  named by the @c href attribute (without extension).  The library must
         *  export @c AeonGUI_OnLoad and optionally @c AeonGUI_OnUnload.
         *
         *  @par Example SVG usage
         *  @code
         *  <script type="native" href="button"/>
         *  @endcode
         *
         *  This loads @c button.dll (Windows), and on Unix-like platforms tries
         *  both @c libbutton.so/@c libbutton.dylib and @c button.so/@c button.dylib.
         *  It then calls @c AeonGUI_OnLoad with a context that exposes the DOM API.
         *
         *  @see PluginAPI.h
         */
        class SVGScriptElement : public SVGElement
        {
        public:
            /** @brief Construct an SVGScriptElement.
             *  @param aTagName    Tag name ("script").
             *  @param aAttributes Element attributes.
             *  @param aParent     Parent node.
             */
            SVGScriptElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );

            /** @brief Destructor.  Unloads the plugin if loaded. */
            ~SVGScriptElement() override;

            /** @brief Not drawable.
             *  @return Always false.
             */
            bool IsDrawEnabled() const override;

            /** @brief Called after document load — loads the native plugin. */
            void OnLoad() override;

            /** @brief Called before document unload — unloads the native plugin. */
            void OnUnload() override;

        private:
            /** @brief Build the full library path from the href attribute.
             *  @param aHref The library base name.
             *  @param aCount Output: number of valid candidates.
             *  @return Platform-specific library path candidates (max 2).
             */
            std::array<std::string, 2> BuildLibraryPaths ( const std::string& aHref, size_t& aCount ) const;

            /** @brief Load a shared library by path.
             *  @param aPath Full path to the library.
             */
            void LoadLibrary ( const std::string& aPath );

            /** @brief Unload the currently loaded shared library. */
            void UnloadLibrary();

            // Static C API callbacks (implementations delegate to C++ DOM)
            static AeonGUI_Element* API_getElementById ( AeonGUI_Document* doc, const char* id );
            static void API_addEventListener ( AeonGUI_Element* element, const char* type, AeonGUI_EventCallback callback, void* userData );
            static void API_removeEventListener ( AeonGUI_Element* element, const char* type, AeonGUI_EventCallback callback, void* userData );
            static const char* API_getAttribute ( AeonGUI_Element* element, const char* name );
            static void API_setAttribute ( AeonGUI_Element* element, const char* name, const char* value );
            static const char* API_getEventType ( AeonGUI_Event* event );
            static AeonGUI_Element* API_querySelector ( AeonGUI_Element* element, const char* selector );

            void* mLibHandle {nullptr};
            AeonGUI_OnLoadFunc mOnLoadFunc {nullptr};
            AeonGUI_OnUnloadFunc mOnUnloadFunc{nullptr};
            AeonGUI_PluginContext mContext{};

            /// Registered callback bridges for cleanup.
            struct RegisteredCallback
            {
                Element* element;
                std::string eventType;
                std::unique_ptr<CallbackEventListener> listener;
            };
            std::vector<RegisteredCallback> mCallbacks;

            /// Weak back-pointer for static API callbacks to find the script element.
            static thread_local SVGScriptElement* sActiveScriptElement;
        };
    }
}
#endif
