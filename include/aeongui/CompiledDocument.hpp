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
#ifndef AEONGUI_COMPILEDDOCUMENT_H
#define AEONGUI_COMPILEDDOCUMENT_H
#include <functional>
#include <string>
#include <unordered_map>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Window;
        class Document;
    }

    /** @brief Abstract base for documents whose DOM tree and scripts are
     *         compiled to C++ at build time by the @c xmlcxx tool.
     *
     *  A CompiledDocument is a "black box": the host owns a @ref DOM::Window
     *  and drives all rendering, resizing and input on it as usual. Loading a
     *  compiled document into that window is done with
     *  @ref DOM::Window::Load(CompiledDocument&), which builds the document's
     *  tree (via @ref BuildDOM) into the window's existing document and then
     *  runs @ref OnLoad.
     *
     *  Communication with the host is intentionally minimal and id-agnostic:
     *  the host registers named callbacks with @ref SetCallback and exchanges
     *  string properties with @ref SetProperty / @ref GetProperty. The
     *  document's compiled C++ raises those callbacks with @ref Emit. The host
     *  never needs to know about element ids, hover, click, etc.
     *
     *  The generator emits one concrete subclass per input file (e.g.
     *  @c fps.xhtml produces @c FpsDocument).
     */
    class CompiledDocument
    {
    public:
        /** @brief Type of a host callback. Receives the optional string detail
         *         passed to @ref Emit. */
        using Callback = std::function<void ( const std::string& ) >;

        AEONGUI_DLL CompiledDocument();
        AEONGUI_DLL virtual ~CompiledDocument();

        /**Host API @{*/
        /** @brief Register (or replace) a host callback for a named event.
         *  @param aName     The event name (e.g. "exit").
         *  @param aCallback The function to invoke when the document emits it.
         */
        AEONGUI_DLL void SetCallback ( const std::string& aName, Callback aCallback );
        /** @brief Set a string property exchanged with the document.
         *  @param aKey   Property name.
         *  @param aValue Property value.
         */
        AEONGUI_DLL void SetProperty ( const std::string& aKey, const std::string& aValue );
        /** @brief Get a string property.
         *  @param aKey Property name.
         *  @return The property value, or an empty string if unset.
         */
        AEONGUI_DLL std::string GetProperty ( const std::string& aKey ) const;
        /** @brief The window this document is bound to.
         *  @return The bound window, or nullptr before the document is loaded.
         */
        AEONGUI_DLL DOM::Window* window() const;
        /** @brief The document this object built its tree into.
         *  @return The bound document, or nullptr before the document is loaded.
         */
        AEONGUI_DLL DOM::Document* document() const;
        /**@}*/

        /**Guest API (callable from generated free functions, which receive the
         * document as their first argument) @{*/
        /** @brief Raise a named event towards the host.
         *
         *  Invokes the matching @ref SetCallback callback (if any) with
         *  @p aDetail and also dispatches a real DOM @ref DOM::Event of the
         *  same name on the bound document, so host code may alternatively
         *  subscribe via @c addEventListener.
         *  @param aName   The event name (e.g. "exit").
         *  @param aDetail Optional string payload for the callback.
         */
        AEONGUI_DLL void Emit ( const std::string& aName, const std::string& aDetail = {} );
        /**@}*/

    protected:
        /** @brief Build the document's DOM tree. Implemented by generated code.
         *  @param aDocument The window's document to populate.
         */
        virtual void BuildDOM ( DOM::Document& aDocument ) = 0;
        /** @brief Run once after the tree is built and finalized.
         *
         *  The generated override calls each function named by a @c <body>
         *  (or root element) @c onload="Fn" attribute, as
         *  @c Fn(*this, window, document). Those functions are defined verbatim
         *  in the document's @c text/c++ @c <script> blocks.
         *  @param aDocument The bound document.
         *  @param aWindow   The bound window.
         */
        virtual void OnLoad ( DOM::Document& aDocument, DOM::Window& aWindow )
        {
            ( void ) aDocument;
            ( void ) aWindow;
        }

    private:
        friend class DOM::Window;
        DOM::Window* mWindow{nullptr};
        DOM::Document* mDocument{nullptr};
        std::unordered_map<std::string, Callback> mCallbacks{};
        std::unordered_map<std::string, std::string> mProperties{};
    };
}
#endif
