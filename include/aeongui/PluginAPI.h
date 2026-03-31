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
 * @file PluginAPI.h
 * @brief C API for AeonGUI native plugins.
 *
 * Native plugins are shared libraries loaded via the SVG
 * @c \<script type="native" href="name"/\> element.
 *
 * A plugin must export at least @c AeonGUI_OnLoad.
 * Optionally it may also export @c AeonGUI_OnUnload for cleanup.
 *
 * @code
 * #include <aeongui/PluginAPI.h>
 *
 * static AeonGUI_PluginContext* gCtx = NULL;
 *
 * static void OnButtonClick(AeonGUI_Event* event, void* userData)
 * {
 *     // Handle click
 * }
 *
 * AEONGUI_PLUGIN_EXPORT void AeonGUI_OnLoad(AeonGUI_PluginContext* ctx)
 * {
 *     gCtx = ctx;
 *     AeonGUI_Element* btn = ctx->getElementById(ctx->document, "myButton");
 *     if (btn)
 *     {
 *         ctx->addEventListener(btn, "click", OnButtonClick, NULL);
 *     }
 * }
 *
 * AEONGUI_PLUGIN_EXPORT void AeonGUI_OnUnload(AeonGUI_PluginContext* ctx)
 * {
 *     // Cleanup
 * }
 * @endcode
 */
#ifndef AEONGUI_PLUGIN_API_H
#define AEONGUI_PLUGIN_API_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Export macro for plugin entry points. */
#ifdef _WIN32
#define AEONGUI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define AEONGUI_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/** @brief Opaque handle to a Document. */
typedef struct AeonGUI_Document_s AeonGUI_Document;
/** @brief Opaque handle to an Element. */
typedef struct AeonGUI_Element_s AeonGUI_Element;
/** @brief Opaque handle to an Event. */
typedef struct AeonGUI_Event_s AeonGUI_Event;

/** @brief Callback type for event listeners.
 *  @param event   The event that was dispatched.
 *  @param userData Opaque pointer passed during registration.
 */
typedef void ( *AeonGUI_EventCallback ) ( AeonGUI_Event* event, void* userData );

/**
 * @brief Plugin context passed to OnLoad and OnUnload.
 *
 * Contains the document handle and function pointers for
 * interacting with the DOM.  The plugin does not need to link
 * against AeonGUI — all calls go through these pointers.
 */
typedef struct AeonGUI_PluginContext
{
    /** @brief The document that loaded this plugin. */
    AeonGUI_Document* document;

    /** @brief Find an element by its @c id attribute.
     *  @param doc The document handle.
     *  @param id  The id string to search for.
     *  @return Element handle, or NULL if not found.
     */
    AeonGUI_Element* ( *getElementById ) ( AeonGUI_Document* doc, const char* id );

    /** @brief Register an event listener on an element.
     *  @param element  The target element.
     *  @param type     Event type (e.g. "click", "mouseenter").
     *  @param callback Function to call when the event fires.
     *  @param userData Opaque pointer forwarded to the callback.
     */
    void ( *addEventListener ) ( AeonGUI_Element* element, const char* type, AeonGUI_EventCallback callback, void* userData );

    /** @brief Remove a previously registered event listener.
     *  @param element  The target element.
     *  @param type     Event type.
     *  @param callback The callback to remove.
     *  @param userData The same userData used during registration.
     */
    void ( *removeEventListener ) ( AeonGUI_Element* element, const char* type, AeonGUI_EventCallback callback, void* userData );

    /** @brief Read an attribute value from an element.
     *  @param element The element.
     *  @param name    Attribute name.
     *  @return The attribute value string, or NULL if not found.
     *          The pointer is valid until the element is destroyed.
     */
    const char* ( *getAttribute ) ( AeonGUI_Element* element, const char* name );

    /** @brief Get the type string of an event.
     *  @param event The event handle.
     *  @return The event type (e.g. "click"). Valid for the
     *          lifetime of the event.
     */
    const char* ( *getEventType ) ( AeonGUI_Event* event );

    /** @brief Set (or add) an attribute on an element.
     *  @param element The element.
     *  @param name    Attribute name.
     *  @param value   New value.
     */
    void ( *setAttribute ) ( AeonGUI_Element* element, const char* name, const char* value );
} AeonGUI_PluginContext;

/**
 * @brief Plugin entry point — called after the document is loaded.
 *
 * The plugin must export this symbol.  Use the context to find
 * elements and register event listeners.
 *
 * @param ctx Plugin context (owned by AeonGUI, do not free).
 */
typedef void ( *AeonGUI_OnLoadFunc ) ( AeonGUI_PluginContext* ctx );

/**
 * @brief Plugin cleanup — called before the document is unloaded.
 *
 * Optional.  If exported, AeonGUI calls it to let the plugin
 * release resources before listeners are removed.
 *
 * @param ctx Same context that was passed to OnLoad.
 */
typedef void ( *AeonGUI_OnUnloadFunc ) ( AeonGUI_PluginContext* ctx );

#ifdef __cplusplus
}
#endif

#endif /* AEONGUI_PLUGIN_API_H */
