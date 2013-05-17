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
#include <gl/gl.h>
#include <gl/glu.h>
#include <map>
#include <cassert>
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
    Window ();
    ~Window();
    bool Initialize ( HINSTANCE hInstance );
    void Finalize ( );
    LRESULT OnSize ( WPARAM type, WORD newwidth, WORD newheight );
    LRESULT OnPaint();
    static LRESULT CALLBACK WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
    static void Register ( HINSTANCE hInstance );
    void RenderLoop();
private:
    HWND hWnd;
    HDC hDC;
    HGLRC hRC;
    PIXELFORMATDESCRIPTOR pfd;
    static ATOM atom;
    AeonGUI::OpenGLRenderer renderer;
    AeonGUI::Image* image;
    AeonGUI::Font* font;
    AeonGUI::MainWindow* window;
    float rotation;
    int width;
    int height;
};

ATOM Window::atom = 0;

Window::Window ()
{}
Window::~Window ()
{}

bool Window::Initialize ( HINSTANCE hInstance )
{
    width  = 640;
    height = 480;
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
                            "WGL31", "AeonGUI Windows OpenGL Demo",
                            WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                            0, 0, // Location
                            rect.right - rect.left, rect.bottom - rect.top, // dimensions
                            NULL,
                            NULL,
                            hInstance,
                            NULL );
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
                WGL_CONTEXT_MINOR_VERSION_ARB, 1,
                0
            };

            wglCreateContextAttribsARB = ( PFNWGLCREATECONTEXTATTRIBSARBPROC ) wglGetProcAddress ( "wglCreateContextAttribsARB" );
            wglMakeCurrent ( hDC, NULL );
            wglDeleteContext ( hRC );
            hRC = wglCreateContextAttribsARB ( hDC, NULL, ctxAttribs );
            wglMakeCurrent ( hDC, hRC );
        }
    }
    //---OpenGL 3.0 Context---//
    ShowWindow ( hWnd, SW_SHOW );
    glShadeModel ( GL_SMOOTH );
    glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
    glClearDepth ( 1.0f );
    glHint ( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glViewport ( 0, 0, width, height );
    glMatrixMode ( GL_PROJECTION );
    glLoadIdentity();
    glFrustum ( -1.33f, 1.33f, -1, 1, 3, 10 );
    // switch to left handed system
    glScalef ( 1, 1, -1 );
#if 0
    glGetFloatv ( GL_PROJECTION_MATRIX, m );
    for ( size_t i = 0; i < 4; ++i )
    {
        for ( size_t j = 0; j < 4; ++j )
        {
            std::cout << std::fixed << m[i + j * 4] << " ";
        }
        std::cout << std::endl;
    }
#endif
    glMatrixMode ( GL_MODELVIEW );
    glLoadIdentity();
    glShadeModel ( GL_SMOOTH );
    glClearDepth ( 1.0f );
    glEnable ( GL_DEPTH_TEST );
    glDepthFunc ( GL_LEQUAL );
    glPointSize ( 16 );
    rotation = 0.0f;
    window = new AeonGUI::MainWindow ();
    image = new AeonGUI::Image ( logo_name, logo_width, logo_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, logo_data );
    font = new AeonGUI::Font ( Vera.data, Vera.size );
    if ( !renderer.Initialize ( GetSystemMetrics ( SM_CXSCREEN ), GetSystemMetrics ( SM_CYSCREEN ) ) )
    {
        return false;
    }
    renderer.SetFont ( font );
    std::wstring hello ( L"Hello World" );
    window->SetCaption ( hello );
    return true;
}

void Window::Finalize ( )
{
    DestroyWindow ( hWnd );
    if ( font != NULL )
    {
        delete font;
        font = NULL;
    }
    if ( hRC != NULL )
    {
        wglMakeCurrent ( hDC, NULL );
        wglDeleteContext ( hRC );
        hRC = NULL;
    }
}

void Window::RenderLoop()
{
    const AeonGUI::Color color ( 0xFFFFFFFF );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
#if 0
    glLoadIdentity();
    // Move to
    glTranslatef ( 0.0f, 0.0f, 7.0f );
    // Rotate around current pos
    glRotatef ( rotation += 0.60f, 1.0f, 1.0f, 1.0f );
    if ( rotation >= 360.0f )
    {
        rotation = 0;
    }
    DrawCube ( 1.0f );
#endif
    renderer.BeginRender();
    window->Render ( &renderer );
    renderer.DrawImage ( color, width - logo_width, height - logo_height, image );
    renderer.EndRender();
    SwapBuffers ( hDC );
}

void Window::Register ( HINSTANCE hInstance )
{
    DWORD error = 0;
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
    wcex.lpszClassName = "WGL31";
    wcex.hIconSm = NULL;
    Window::atom = RegisterClassEx ( &wcex );
}

LRESULT CALLBACK Window::WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    LRESULT lresult;
    Window* window_ptr = ( ( Window* ) GetWindowLongPtr ( hwnd, 0 ) );
    switch ( uMsg )
    {
    case WM_PAINT:
        lresult = window_ptr->OnPaint();
        break;
    case WM_CLOSE:
        PostQuitMessage ( 0 );
        lresult = 0;
    case WM_SIZE:
        lresult = window_ptr->OnSize ( wParam, LOWORD ( lParam ), HIWORD ( lParam ) );
        break;
    default:
        lresult = DefWindowProc ( hwnd, uMsg, wParam, lParam );
    }
    return lresult;
}

LRESULT Window::OnSize ( WPARAM type, WORD newwidth, WORD newheight )
{
    width = newwidth;
    height = newheight;
    if ( height == 0 )
    {
        height = 1;
    }
    if ( width == 0 )
    {
        width = 1;
    }
    glViewport ( 0, 0, width, height );
    glMatrixMode ( GL_PROJECTION );
    glLoadIdentity();
    glFrustum ( -1.33f, 1.33f, -1, 1, 3, 10 );
    // switch to left handed system
    glScalef ( 1, 1, -1 );
    glMatrixMode ( GL_MODELVIEW );
    glLoadIdentity();
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

int WINAPI WinMain ( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
    Window window;
    if ( !window.Initialize ( hInstance ) )
    {
        return 0;
    }
    MSG msg;
    memset ( &msg, 0, sizeof ( MSG ) );
    float rotation = 0.0f;
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
