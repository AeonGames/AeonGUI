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
 * @file test_plugin.c
 * @brief Test plugin for SVGScriptElement unit tests.
 *
 * This plugin sets global flags when OnLoad/OnUnload are called
 * and registers an event listener that increments a counter.
 * The test reads the flags/counters via exported getter functions.
 */
#include "aeongui/PluginAPI.h"

static int gLoadCalled = 0;
static int gUnloadCalled = 0;
static int gClickCount = 0;
static AeonGUI_PluginContext* gCtx = 0;

static void OnClick ( AeonGUI_Event* event, void* userData )
{
    ( void ) event;
    ( void ) userData;
    ++gClickCount;
}

AEONGUI_PLUGIN_EXPORT void AeonGUI_OnLoad ( AeonGUI_PluginContext* ctx )
{
    gCtx = ctx;
    gLoadCalled = 1;
    gClickCount = 0;

    AeonGUI_Element* btn = ctx->getElementById ( ctx->document, "testBtn" );
    if ( btn )
    {
        ctx->addEventListener ( btn, "click", OnClick, 0 );
    }
}

AEONGUI_PLUGIN_EXPORT void AeonGUI_OnUnload ( AeonGUI_PluginContext* ctx )
{
    ( void ) ctx;
    gUnloadCalled = 1;
}

AEONGUI_PLUGIN_EXPORT int TestPlugin_GetLoadCalled ( void )
{
    return gLoadCalled;
}

AEONGUI_PLUGIN_EXPORT int TestPlugin_GetUnloadCalled ( void )
{
    return gUnloadCalled;
}

AEONGUI_PLUGIN_EXPORT int TestPlugin_GetClickCount ( void )
{
    return gClickCount;
}
