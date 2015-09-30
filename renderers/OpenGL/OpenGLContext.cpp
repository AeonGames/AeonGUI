/******************************************************************************
Copyright 2015 Rodrigo Hernandez Cordoba

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

#include <windows.h>
#include "OpenGL.h"
#include "wglext.h"

namespace AeonGUI
{
    static HWND hWnd;
    static HDC hDC;
    static HGLRC hRC;
    static ATOM atom;

    bool CreateOpenGLContext()
    {
        PIXELFORMATDESCRIPTOR pfd;
        RECT rect = { 0, 0, 10, 10 };
        WNDCLASSEX wcex;
        wcex.cbSize = sizeof ( WNDCLASSEX );
        wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wcex.lpfnWndProc = ( WNDPROC ) DefWindowProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = 0;
        wcex.hInstance = GetModuleHandle ( NULL );
        wcex.hIcon = LoadIcon ( NULL, IDI_WINLOGO );
        wcex.hCursor = LoadCursor ( NULL, IDC_ARROW );
        wcex.hbrBackground = NULL;
        wcex.lpszMenuName = NULL;
        wcex.lpszClassName = "glUnitTest";
        wcex.hIconSm = NULL;
        atom = RegisterClassEx ( &wcex );
        hWnd = CreateWindowEx ( WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
                                MAKEINTATOM ( atom ), "OpenGL Unit Testing Window",
                                WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                                0, 0, // Location
                                rect.right - rect.left, rect.bottom - rect.top, // dimensions
                                NULL,
                                NULL,
                                GetModuleHandle ( NULL ),
                                NULL );
        hDC = GetDC ( hWnd );
        pfd.nSize = sizeof ( PIXELFORMATDESCRIPTOR );
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        pfd.cRedBits = 0;
        pfd.cRedShift = 0;
        pfd.cGreenBits = 0;
        pfd.cGreenShift = 0;
        pfd.cBlueBits = 0;
        pfd.cBlueShift = 0;
        pfd.cAlphaBits = 0;
        pfd.cAlphaShift = 0;
        pfd.cAccumBits = 0;
        pfd.cAccumRedBits = 0;
        pfd.cAccumGreenBits = 0;
        pfd.cAccumBlueBits = 0;
        pfd.cAccumAlphaBits = 0;
        pfd.cDepthBits = 32;
        pfd.cStencilBits = 0;
        pfd.cAuxBuffers = 0;
        pfd.iLayerType = PFD_MAIN_PLANE;
        pfd.bReserved = 0;
        pfd.dwLayerMask = 0;
        pfd.dwVisibleMask = 0;
        pfd.dwDamageMask = 0;
        int pf = ChoosePixelFormat ( hDC, &pfd );
        SetPixelFormat ( hDC, pf, &pfd );
        hRC = wglCreateContext ( hDC );
        wglMakeCurrent ( hDC, hRC );
        return true;
    }

    void DestroyOpenGLContext()
    {
        wglMakeCurrent ( hDC, NULL );
        wglDeleteContext ( hRC );
        ReleaseDC ( hWnd, hDC );
        DestroyWindow ( hWnd );
        UnregisterClass ( reinterpret_cast<LPCSTR> (
#if defined(_M_X64) || defined(__amd64__)
                              0x0ULL +
#endif
                              MAKELONG ( atom, 0 ) ), NULL );
    }
}