/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

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
#include "MainWindow.h"
#include "glcommon.h"
#include "logo.h"
#include "Vera.h"
#include "Color.h"

class Window
{
public:
    Window() {};
    ~Window() {};
    void Initialize ( HINSTANCE hInstance );
    void Finalize ( );
    LRESULT OnSize ( WPARAM type, WORD newwidth, WORD newheight );
    LRESULT OnPaint();
    LRESULT OnMouseMove ( int32_t x, int32_t y );
    static LRESULT CALLBACK WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
    static void Register ( HINSTANCE hInstance );
    void RenderLoop();
private:
    static ATOM atom;
    HWND hWnd;
    HDC hDC;
    HGLRC hRC;
    PIXELFORMATDESCRIPTOR pfd;
    int32_t width;
    int32_t height;
    int32_t mousex;
    int32_t mousey;
    AeonGUI::OpenGLRenderer renderer;
    AeonGUI::Image* image;
    AeonGUI::Font* font;
    AeonGUI::MainWindow* window;
};

ATOM Window::atom = 0;

void Window::Initialize ( HINSTANCE hInstance )
{
    width  = 800;
    height = 600;
    int pf;

    RECT rect = {0, 0, width, height};

    PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = NULL;
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = NULL;

    if ( atom == 0 )
    {
        Register ( hInstance );
    }
    AdjustWindowRectEx ( &rect, WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, FALSE, WS_EX_APPWINDOW | WS_EX_WINDOWEDGE );
    hWnd = CreateWindowEx ( WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
                            "AeonGUI", "AeonGUI",
                            WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                            0, 0, // Location
                            rect.right - rect.left, rect.bottom - rect.top, // dimensions
                            NULL,
                            NULL,
                            hInstance,
                            this );
    SetWindowLongPtr ( hWnd, 0, ( LONG_PTR ) this );
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
    pf = ChoosePixelFormat ( hDC, &pfd );
    SetPixelFormat ( hDC, pf, &pfd );
    hRC = wglCreateContext ( hDC );
    wglMakeCurrent ( hDC, hRC );

    //---OpenGL 3.0 Context---//
    wglGetExtensionsStringARB = ( PFNWGLGETEXTENSIONSSTRINGARBPROC ) wglGetProcAddress ( "wglGetExtensionsStringARB" );
    if ( wglGetExtensionsStringARB != NULL )
    {
        if ( strstr ( wglGetExtensionsStringARB ( hDC ), "WGL_ARB_create_context" ) != NULL )
        {
            const int ctxAttribs[] =
            {
                WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
                WGL_CONTEXT_MINOR_VERSION_ARB, 2,
                WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                0
            };

            wglCreateContextAttribsARB = ( PFNWGLCREATECONTEXTATTRIBSARBPROC ) wglGetProcAddress ( "wglCreateContextAttribsARB" );
            wglMakeCurrent ( NULL, NULL );
            wglDeleteContext ( hRC );
            hRC = wglCreateContextAttribsARB ( hDC, NULL, ctxAttribs );
            wglMakeCurrent ( hDC, hRC );
        }
    }
    //---OpenGL 3.0 Context---//
    window = new AeonGUI::MainWindow ();
    image = new AeonGUI::Image ( logo_name, logo_width, logo_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, logo_data );
    font = new AeonGUI::Font ( Vera.data, Vera.size );
    renderer.Initialize ( GetSystemMetrics ( SM_CXSCREEN ), GetSystemMetrics ( SM_CYSCREEN ) );
    renderer.SetFont ( font );
    std::wstring hello ( L"Hello World" );
    window->SetCaption ( hello );
    ShowWindow ( hWnd, SW_SHOW );
}

void Window::Finalize()
{
    wglMakeCurrent ( hDC, NULL );
    wglDeleteContext ( hRC );
    ReleaseDC ( hWnd, hDC );
    DestroyWindow ( hWnd );
}

void Window::RenderLoop()
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER this_time;
    QueryPerformanceCounter ( &this_time );
    QueryPerformanceFrequency ( &frequency );
    static LARGE_INTEGER last_time = this_time;
    float delta = static_cast<float> ( this_time.QuadPart - last_time.QuadPart ) / static_cast<float> ( frequency.QuadPart );
    if ( delta > 1e-1f )
    {
        delta = 1.0f / 30.0f;
    }

    const AeonGUI::Color color ( 0xFFFFFFFF );
    renderer.BeginRender();
    window->Render ( &renderer );
    renderer.DrawImage ( color, width - logo_width, height - logo_height, image );
    renderer.EndRender();

    SwapBuffers ( hDC );
    last_time = this_time;
}

void Window::Register ( HINSTANCE hInstance )
{
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof ( WNDCLASSEX );
    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wcex.lpfnWndProc = ( WNDPROC ) Window::WindowProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = sizeof ( Window* );
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon ( NULL, IDI_WINLOGO );
    wcex.hCursor = LoadCursor ( NULL, IDC_ARROW );
    wcex.hbrBackground = NULL;
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = "AeonGUI";
    wcex.hIconSm = NULL;
    Window::atom = RegisterClassEx ( &wcex );
}

LRESULT CALLBACK Window::WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    LRESULT lresult = 0;
    Window* window_ptr = ( ( Window* ) GetWindowLongPtr ( hwnd, 0 ) );
    switch ( uMsg )
    {
    case WM_PAINT:
        lresult = window_ptr->OnPaint();
        break;
    case WM_CLOSE:
        PostQuitMessage ( 0 );
    case WM_SIZE:
        lresult = window_ptr->OnSize ( wParam, LOWORD ( lParam ), HIWORD ( lParam ) );
        break;
    case WM_KEYDOWN:
        lresult = DefWindowProc ( hwnd, uMsg, wParam, lParam );
        break;
    case WM_KEYUP:
        lresult = DefWindowProc ( hwnd, uMsg, wParam, lParam );
        break;
    case WM_MOUSEMOVE:
        lresult = window_ptr->OnMouseMove ( GET_X_LPARAM ( lParam ), GET_Y_LPARAM ( lParam ) );
        break;
    case WM_LBUTTONDOWN:
        break;
    case WM_LBUTTONUP:
        break;
    default:
        lresult = DefWindowProc ( hwnd, uMsg, wParam, lParam );
    }
    return lresult;
}

LRESULT Window::OnSize ( WPARAM type, WORD newwidth, WORD newheight )
{
    width = static_cast<int32_t> ( newwidth );
    height = static_cast<int32_t> ( newheight );
    if ( height == 0 )
    {
        height = 1;
    }
    if ( width == 0 )
    {
        width = 1;
    }
    return 0;
}

LRESULT Window::OnPaint()
{
    RECT rect;
    PAINTSTRUCT paint;
    if ( GetUpdateRect ( hWnd, &rect, FALSE ) )
    {
        BeginPaint ( hWnd, &paint );
        EndPaint ( hWnd, &paint );
    }
    return 0;
}

LRESULT Window::OnMouseMove ( int32_t x, int32_t y )
{
    return 0;
}

int WINAPI WinMain ( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
    Window window;
    window.Initialize ( hInstance );
    MSG msg;
    memset ( &msg, 0, sizeof ( MSG ) );
    while ( msg.message != WM_QUIT )
    {
        if ( PeekMessage ( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            if ( msg.message != WM_QUIT )
            {
                TranslateMessage ( &msg );
                DispatchMessage ( &msg );
            }
        }
        else
        {
            window.RenderLoop();
        }
    }
    assert ( msg.message == WM_QUIT );
    window.Finalize();
    return static_cast<int> ( msg.wParam );
}

int main ( int argc, char *argv[] )
{
#ifdef _MSC_VER
    // Send all reports to STDOUT
    _CrtSetReportMode ( _CRT_WARN, _CRTDBG_MODE_FILE );
    _CrtSetReportFile ( _CRT_WARN, _CRTDBG_FILE_STDOUT );
    _CrtSetReportMode ( _CRT_ERROR, _CRTDBG_MODE_FILE );
    _CrtSetReportFile ( _CRT_ERROR, _CRTDBG_FILE_STDOUT );
    _CrtSetReportMode ( _CRT_ASSERT, _CRTDBG_MODE_FILE );
    _CrtSetReportFile ( _CRT_ASSERT, _CRTDBG_FILE_STDOUT );
    // Use _CrtSetBreakAlloc( ) to set breakpoints on allocations.
#endif
    int ret = WinMain ( GetModuleHandle ( NULL ), NULL, NULL, 0 );
#ifdef _MSC_VER
    _CrtDumpMemoryLeaks();
#endif
    return ret;
}
