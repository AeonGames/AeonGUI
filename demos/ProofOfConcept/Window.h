/******************************************************************************
Copyright 2010-2012,2015 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#ifndef AEONGUI_POC_WINDOW_H
#define AEONGUI_POC_WINDOW_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <crtdbg.h>
#include "wglext.h"
#include "OpenGLRenderer.h"

class Window
{
public:
    Window ( HINSTANCE hInstance, LONG aWidth, LONG aHeight );
    ~Window();
    LRESULT OnSize ( WPARAM type, WORD newwidth, WORD newheight );
    LRESULT OnPaint();
    LRESULT OnMouseMove ( int32_t x, int32_t y );
    LRESULT OnMouseButtonDown ( uint8_t button, int32_t x, int32_t y );
    LRESULT OnMouseButtonUp ( uint8_t button, int32_t x, int32_t y );
    void RenderLoop();
    static LRESULT CALLBACK WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
    static void Register ( HINSTANCE hInstance );
private:
    static ATOM atom;
    HWND hWnd;
    HDC hDC;
    HGLRC hRC;
    AeonGUI::OpenGLRenderer GUIRenderer;
};
#endif