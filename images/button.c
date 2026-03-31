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
 * @file button.c
 * @brief Native plugin implementation for button.svg.
 *
 * Replaces the JavaScript event handlers with C callbacks
 * loaded via @c \<script type="native" href="button"/\>.
 */
#include "aeongui/PluginAPI.h"
#include <stdio.h>
#include <string.h>

static AeonGUI_PluginContext* gCtx = NULL;

/* ---- Button descriptor ---- */

typedef struct
{
    const char* groupId;
    const char* rectId;
    const char* label;
    const char* normalGrad;
    const char* hoverGrad;
    const char* activeGrad;
} ButtonInfo;

static const ButtonInfo gButtons[] =
{
    { "blueBtn",  "blueBtnRect",  "Blue Button",  "url(#btnGrad)",      "url(#btnGradHover)",      "url(#btnGradActive)" },
    { "greenBtn", "greenBtnRect", "Green Button", "url(#btnGradGreen)", "url(#btnGradGreenHover)", "url(#btnGradGreenActive)" },
    { "redBtn",   "redBtnRect",   "Red Button",   "url(#btnGradRed)",   "url(#btnGradRedHover)",   "url(#btnGradRedActive)" },
};

#define BUTTON_COUNT (sizeof(gButtons) / sizeof(gButtons[0]))

/* ---- Toggle state ---- */

static int gToggleOn = 0;

/* ---- Helpers ---- */

static void setStatus ( const char* text )
{
    AeonGUI_Element* status = gCtx->getElementById ( gCtx->document, "status" );
    if ( status )
    {
        gCtx->setAttribute ( status, "textContent", text );
    }
}

/* ---- Color button callbacks ---- */

static void onMouseOver ( AeonGUI_Event* event, void* userData )
{
    const ButtonInfo* info = ( const ButtonInfo* ) userData;
    AeonGUI_Element* rect = gCtx->getElementById ( gCtx->document, info->rectId );
    if ( rect )
    {
        gCtx->setAttribute ( rect, "fill", info->hoverGrad );
    }
    char msg[64];
    snprintf ( msg, sizeof ( msg ), "Mouse over: %s", info->label );
    setStatus ( msg );

    ( void ) event;
}

static void onMouseOut ( AeonGUI_Event* event, void* userData )
{
    const ButtonInfo* info = ( const ButtonInfo* ) userData;
    AeonGUI_Element* rect = gCtx->getElementById ( gCtx->document, info->rectId );
    AeonGUI_Element* group = gCtx->getElementById ( gCtx->document, info->groupId );
    if ( rect )
    {
        gCtx->setAttribute ( rect, "fill", info->normalGrad );
    }
    if ( group )
    {
        gCtx->setAttribute ( group, "filter", "url(#shadow)" );
        gCtx->setAttribute ( group, "transform", "" );
    }
    char msg[64];
    snprintf ( msg, sizeof ( msg ), "Mouse left: %s", info->label );
    setStatus ( msg );

    ( void ) event;
}

static void onMouseDown ( AeonGUI_Event* event, void* userData )
{
    const ButtonInfo* info = ( const ButtonInfo* ) userData;
    AeonGUI_Element* rect = gCtx->getElementById ( gCtx->document, info->rectId );
    AeonGUI_Element* group = gCtx->getElementById ( gCtx->document, info->groupId );
    if ( rect )
    {
        gCtx->setAttribute ( rect, "fill", info->activeGrad );
    }
    if ( group )
    {
        gCtx->setAttribute ( group, "filter", "url(#shadowPressed)" );
        gCtx->setAttribute ( group, "transform", "translate(1,1)" );
    }
    char msg[64];
    snprintf ( msg, sizeof ( msg ), "Mouse down: %s", info->label );
    setStatus ( msg );

    ( void ) event;
}

static void onMouseUp ( AeonGUI_Event* event, void* userData )
{
    const ButtonInfo* info = ( const ButtonInfo* ) userData;
    AeonGUI_Element* rect = gCtx->getElementById ( gCtx->document, info->rectId );
    AeonGUI_Element* group = gCtx->getElementById ( gCtx->document, info->groupId );
    if ( rect )
    {
        gCtx->setAttribute ( rect, "fill", info->hoverGrad );
    }
    if ( group )
    {
        gCtx->setAttribute ( group, "filter", "url(#shadow)" );
        gCtx->setAttribute ( group, "transform", "" );
    }
    char msg[64];
    snprintf ( msg, sizeof ( msg ), "Clicked: %s!", info->label );
    setStatus ( msg );

    ( void ) event;
}

/* ---- Toggle button callback ---- */

static void onToggleDown ( AeonGUI_Event* event, void* userData )
{
    AeonGUI_Element* rect = gCtx->getElementById ( gCtx->document, "toggleBtnRect" );
    AeonGUI_Element* text = gCtx->getElementById ( gCtx->document, "toggleBtnText" );

    if ( !gToggleOn )
    {
        if ( rect )
        {
            gCtx->setAttribute ( rect, "fill", "#4a4" );
            gCtx->setAttribute ( rect, "stroke", "#2a2" );
        }
        if ( text )
        {
            gCtx->setAttribute ( text, "textContent", "ON" );
        }
        setStatus ( "Toggle: ON" );
        gToggleOn = 1;
    }
    else
    {
        if ( rect )
        {
            gCtx->setAttribute ( rect, "fill", "#888" );
            gCtx->setAttribute ( rect, "stroke", "#666" );
        }
        if ( text )
        {
            gCtx->setAttribute ( text, "textContent", "OFF" );
        }
        setStatus ( "Toggle: OFF" );
        gToggleOn = 0;
    }

    ( void ) event;
    ( void ) userData;
}

/* ---- Plugin entry points ---- */

AEONGUI_PLUGIN_EXPORT void AeonGUI_OnLoad ( AeonGUI_PluginContext* ctx )
{
    unsigned int i;
    gCtx = ctx;
    gToggleOn = 0;

    for ( i = 0; i < BUTTON_COUNT; ++i )
    {
        AeonGUI_Element* group = ctx->getElementById ( ctx->document, gButtons[i].groupId );
        if ( group )
        {
            ctx->addEventListener ( group, "mouseenter",  onMouseOver, ( void* ) &gButtons[i] );
            ctx->addEventListener ( group, "mouseleave",  onMouseOut,  ( void* ) &gButtons[i] );
            ctx->addEventListener ( group, "mousedown",   onMouseDown, ( void* ) &gButtons[i] );
            ctx->addEventListener ( group, "mouseup",     onMouseUp,   ( void* ) &gButtons[i] );
        }
    }

    AeonGUI_Element* toggle = ctx->getElementById ( ctx->document, "toggleBtn" );
    if ( toggle )
    {
        ctx->addEventListener ( toggle, "mousedown", onToggleDown, NULL );
    }
}

AEONGUI_PLUGIN_EXPORT void AeonGUI_OnUnload ( AeonGUI_PluginContext* ctx )
{
    ( void ) ctx;
    gCtx = NULL;
}
