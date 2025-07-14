/*
Copyright (C) 2010-2012,2019-2021,2023,2025 Rodrigo Jose Hernandez Cordoba

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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <GL/glu.h>
#include <GL/wglext.h>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdint>
#include <crtdbg.h>
#include "aeongui/AeonGUI.h"
#include "aeongui/dom/Window.h"
#include "aeongui/dom/Document.h"

#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
        glFunction = (glFunctionType)wglGetProcAddress(#glFunction); \
        if (glFunction == nullptr) { std::cerr << "OpenGL: Unable to load "<< #glFunction <<" function." << std::endl; }}
#include "Common.h"

GLDEFINEFUNCTION ( PFNGLISPROGRAMPROC, glIsProgram );
GLDEFINEFUNCTION ( PFNGLISSHADERPROC, glIsShader );
GLDEFINEFUNCTION ( PFNGLCREATEPROGRAMPROC, glCreateProgram );
GLDEFINEFUNCTION ( PFNGLCREATESHADERPROC, glCreateShader );
GLDEFINEFUNCTION ( PFNGLCOMPILESHADERPROC, glCompileShader );
GLDEFINEFUNCTION ( PFNGLDETACHSHADERPROC, glDetachShader );
GLDEFINEFUNCTION ( PFNGLDELETESHADERPROC, glDeleteShader );
GLDEFINEFUNCTION ( PFNGLDELETEPROGRAMPROC, glDeleteProgram );
GLDEFINEFUNCTION ( PFNGLATTACHSHADERPROC, glAttachShader );
GLDEFINEFUNCTION ( PFNGLSHADERSOURCEPROC, glShaderSource );
GLDEFINEFUNCTION ( PFNGLUSEPROGRAMPROC, glUseProgram );
GLDEFINEFUNCTION ( PFNGLLINKPROGRAMPROC, glLinkProgram );
GLDEFINEFUNCTION ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
GLDEFINEFUNCTION ( PFNGLBINDVERTEXARRAYPROC, glBindVertexArray );
GLDEFINEFUNCTION ( PFNGLGENBUFFERSPROC, glGenBuffers );
GLDEFINEFUNCTION ( PFNGLBINDBUFFERPROC, glBindBuffer );
GLDEFINEFUNCTION ( PFNGLBUFFERDATAPROC, glBufferData );
GLDEFINEFUNCTION ( PFNGLISBUFFERPROC, glIsBuffer );
GLDEFINEFUNCTION ( PFNGLISVERTEXARRAYPROC, glIsVertexArray );
GLDEFINEFUNCTION ( PFNGLDELETEBUFFERSPROC, glDeleteBuffers );
GLDEFINEFUNCTION ( PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays );
GLDEFINEFUNCTION ( PFNGLACTIVETEXTUREPROC, glActiveTexture );
GLDEFINEFUNCTION ( PFNGLUNIFORM1IPROC, glUniform1i );
GLDEFINEFUNCTION ( PFNGLGETSHADERIVPROC, glGetShaderiv );
GLDEFINEFUNCTION ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
GLDEFINEFUNCTION ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
GLDEFINEFUNCTION ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );

class Window
{
public:
    Window ( HINSTANCE hInstance, LPSTR aFilename, LONG aWidth, LONG aHeight ) :
        mDocument{aFilename ? aFilename : ""}, mWindow{&mDocument, static_cast<uint32_t> ( aWidth ), static_cast<uint32_t> ( aHeight ) }
    {
        Initialize ( hInstance, aWidth, aHeight );
    }
    ~Window()
    {
        Finalize();
    }
    LRESULT OnSize ( WPARAM type, WORD newwidth, WORD newheight );
    LRESULT OnPaint();
    LRESULT OnMouseMove ( int32_t x, int32_t y );
    LRESULT OnMouseButtonDown ( uint8_t button, int32_t x, int32_t y );
    LRESULT OnMouseButtonUp ( uint8_t button, int32_t x, int32_t y );
    static LRESULT CALLBACK WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
    static void Register ( HINSTANCE hInstance );
    void RenderLoop();
private:
    static ATOM atom;
    void Initialize ( HINSTANCE hInstance, LONG aWidth, LONG aHeight );
    void Finalize();
    HWND hWnd{};
    HDC hDC{};
    HGLRC hRC{};
    GLuint mProgram{};
    GLuint mVAO{};
    GLuint mScreenQuad{};
    GLuint mScreenTexture{};
    AeonGUI::Document mDocument{};
    AeonGUI::Window mWindow{};
};

ATOM Window::atom = 0;

void Window::Initialize ( HINSTANCE hInstance, LONG aWidth, LONG aHeight )
{
    std::cout << "Width: " << mWindow.GetWidth() << std::endl;
    std::cout << "Height: " << mWindow.GetHeight() << std::endl;
    std::cout << "Stride: " << mWindow.GetStride() << std::endl;
    int pf{};
    PIXELFORMATDESCRIPTOR pfd{};
    RECT rect{0, 0, aWidth, aHeight};

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
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;
    pfd.iLayerType = PFD_MAIN_PLANE;
    pf = ChoosePixelFormat ( hDC, &pfd );
    SetPixelFormat ( hDC, pf, &pfd );
    hRC = wglCreateContext ( hDC );
    wglMakeCurrent ( hDC, hRC );

    //---OpenGL 4.5 Context---//
    wglGetExtensionsStringARB = ( PFNWGLGETEXTENSIONSSTRINGARBPROC ) wglGetProcAddress ( "wglGetExtensionsStringARB" );
    if ( wglGetExtensionsStringARB != NULL )
    {
        if ( strstr ( wglGetExtensionsStringARB ( hDC ), "WGL_ARB_create_context" ) != NULL )
        {
            const int ctxAttribs[] =
            {
                WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                WGL_CONTEXT_MINOR_VERSION_ARB, 5,
                WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                0
            };

            wglCreateContextAttribsARB = ( PFNWGLCREATECONTEXTATTRIBSARBPROC ) wglGetProcAddress ( "wglCreateContextAttribsARB" );
            wglMakeCurrent ( hDC, NULL );
            wglDeleteContext ( hRC );
            hRC = wglCreateContextAttribsARB ( hDC, NULL, ctxAttribs );
            wglMakeCurrent ( hDC, hRC );
        }
    }
    //---OpenGL 4.5 Context---//

    GLGETPROCADDRESS ( PFNGLISPROGRAMPROC, glIsProgram );
    GLGETPROCADDRESS ( PFNGLISSHADERPROC, glIsShader );
    GLGETPROCADDRESS ( PFNGLCREATEPROGRAMPROC, glCreateProgram );
    GLGETPROCADDRESS ( PFNGLCREATESHADERPROC, glCreateShader );
    GLGETPROCADDRESS ( PFNGLCOMPILESHADERPROC, glCompileShader );
    GLGETPROCADDRESS ( PFNGLDETACHSHADERPROC, glDetachShader );
    GLGETPROCADDRESS ( PFNGLDELETESHADERPROC, glDeleteShader );
    GLGETPROCADDRESS ( PFNGLDELETEPROGRAMPROC, glDeleteProgram );
    GLGETPROCADDRESS ( PFNGLATTACHSHADERPROC, glAttachShader );
    GLGETPROCADDRESS ( PFNGLSHADERSOURCEPROC, glShaderSource );
    GLGETPROCADDRESS ( PFNGLUSEPROGRAMPROC, glUseProgram );
    GLGETPROCADDRESS ( PFNGLLINKPROGRAMPROC, glLinkProgram );
    GLGETPROCADDRESS ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
    GLGETPROCADDRESS ( PFNGLBINDVERTEXARRAYPROC, glBindVertexArray );
    GLGETPROCADDRESS ( PFNGLGENBUFFERSPROC, glGenBuffers );
    GLGETPROCADDRESS ( PFNGLBINDBUFFERPROC, glBindBuffer );
    GLGETPROCADDRESS ( PFNGLBUFFERDATAPROC, glBufferData );
    GLGETPROCADDRESS ( PFNGLISBUFFERPROC, glIsBuffer );
    GLGETPROCADDRESS ( PFNGLISVERTEXARRAYPROC, glIsVertexArray );
    GLGETPROCADDRESS ( PFNGLDELETEBUFFERSPROC, glDeleteBuffers );
    GLGETPROCADDRESS ( PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays );
    GLGETPROCADDRESS ( PFNGLACTIVETEXTUREPROC, glActiveTexture );
    GLGETPROCADDRESS ( PFNGLUNIFORM1IPROC, glUniform1i );
    GLGETPROCADDRESS ( PFNGLGETSHADERIVPROC, glGetShaderiv );
    GLGETPROCADDRESS ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
    GLGETPROCADDRESS ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
    GLGETPROCADDRESS ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );

    glClearColor ( 1, 1, 1, 1 );
    glViewport ( 0, 0, aWidth, aHeight );
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    OPENGL_CHECK_ERROR;
    glEnable ( GL_BLEND );
    OPENGL_CHECK_ERROR;

    GLint compile_status{};
    mProgram = glCreateProgram();
    OPENGL_CHECK_ERROR;
    GLuint vertex_shader = glCreateShader ( GL_VERTEX_SHADER );
    OPENGL_CHECK_ERROR;
    glShaderSource ( vertex_shader, 1, &vertex_shader_code_ptr, &vertex_shader_len );
    OPENGL_CHECK_ERROR;
    glCompileShader ( vertex_shader );
    OPENGL_CHECK_ERROR;
    glGetShaderiv ( vertex_shader, GL_COMPILE_STATUS, &compile_status );
    OPENGL_CHECK_ERROR;
    if ( compile_status != GL_TRUE )
    {
        GLint info_log_len;
        glGetShaderiv ( vertex_shader, GL_INFO_LOG_LENGTH, &info_log_len );
        OPENGL_CHECK_ERROR;
        std::string log_string;
        log_string.resize ( info_log_len );
        if ( info_log_len > 1 )
        {
            glGetShaderInfoLog ( vertex_shader, info_log_len, nullptr, const_cast<GLchar*> ( log_string.data() ) );
            OPENGL_CHECK_ERROR;
            std::cout << vertex_shader_code << std::endl;
            std::cout << log_string << std::endl;
        }
    }
    glAttachShader ( mProgram, vertex_shader );
    OPENGL_CHECK_ERROR
    //-------------------------
    uint32_t fragment_shader = glCreateShader ( GL_FRAGMENT_SHADER );
    OPENGL_CHECK_ERROR
    glShaderSource ( fragment_shader, 1, &fragment_shader_code_ptr, &fragment_shader_len );
    OPENGL_CHECK_ERROR
    glCompileShader ( fragment_shader );
    OPENGL_CHECK_ERROR;
    glGetShaderiv ( fragment_shader, GL_COMPILE_STATUS, &compile_status );
    OPENGL_CHECK_ERROR;
    if ( compile_status != GL_TRUE )
    {
        GLint info_log_len;
        glGetShaderiv ( fragment_shader, GL_INFO_LOG_LENGTH, &info_log_len );
        OPENGL_CHECK_ERROR;
        std::string log_string;
        log_string.resize ( info_log_len );
        if ( info_log_len > 1 )
        {
            glGetShaderInfoLog ( fragment_shader, info_log_len, nullptr, const_cast<GLchar*> ( log_string.data() ) );
            OPENGL_CHECK_ERROR;
            std::cout << fragment_shader_code << std::endl;
            std::cout << log_string << std::endl;
        }
    }
    glAttachShader ( mProgram, fragment_shader );
    OPENGL_CHECK_ERROR;
    //-------------------------
    glLinkProgram ( mProgram );
    OPENGL_CHECK_ERROR;
    glDetachShader ( mProgram, vertex_shader );
    OPENGL_CHECK_ERROR;
    glDetachShader ( mProgram, fragment_shader );
    OPENGL_CHECK_ERROR;
    glDeleteShader ( vertex_shader );
    OPENGL_CHECK_ERROR;
    glDeleteShader ( fragment_shader );
    OPENGL_CHECK_ERROR;
    glUseProgram ( mProgram );
    OPENGL_CHECK_ERROR;
    glUniform1i ( 0, 0 );
    OPENGL_CHECK_ERROR;

    //---------------------------------------------------------------------------
    glGenVertexArrays ( 1, &mVAO );
    OPENGL_CHECK_ERROR;
    glBindVertexArray ( mVAO );
    OPENGL_CHECK_ERROR;
    glGenBuffers ( 1, &mScreenQuad );
    OPENGL_CHECK_ERROR;
    glBindBuffer ( GL_ARRAY_BUFFER, mScreenQuad );
    OPENGL_CHECK_ERROR;
    glBufferData ( GL_ARRAY_BUFFER, vertex_size, vertices, GL_STATIC_DRAW );
    OPENGL_CHECK_ERROR;
    glEnableVertexAttribArray ( 0 );
    OPENGL_CHECK_ERROR;
    glVertexAttribPointer ( 0, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, 0 );
    OPENGL_CHECK_ERROR;
    glEnableVertexAttribArray ( 1 );
    OPENGL_CHECK_ERROR;
    glVertexAttribPointer ( 1, 2, GL_FLOAT, GL_FALSE, sizeof ( float ) * 4, reinterpret_cast<const void*> ( sizeof ( float ) * 2 ) );
    OPENGL_CHECK_ERROR;

    //---------------------------------------------------------------------------
    glGenTextures ( 1, &mScreenTexture );
    OPENGL_CHECK_ERROR;
    glBindTexture ( GL_TEXTURE_2D, mScreenTexture );
    OPENGL_CHECK_ERROR;
    glTexImage2D ( GL_TEXTURE_2D,
                   0,
                   GL_RGBA,
                   aWidth,
                   aHeight,
                   0,
                   GL_BGRA,
                   GL_UNSIGNED_INT_8_8_8_8_REV,
                   mWindow.GetPixels() );
    OPENGL_CHECK_ERROR;
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    OPENGL_CHECK_ERROR;
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    OPENGL_CHECK_ERROR;
    glActiveTexture ( GL_TEXTURE0 );
    OPENGL_CHECK_ERROR;

    ShowWindow ( hWnd, SW_SHOW );
}

void Window::Finalize()
{
    if ( glIsTexture ( mScreenTexture ) )
    {
        glDeleteTextures ( 1, &mScreenTexture );
        mScreenTexture = 0;
    }
    if ( glIsBuffer ( mScreenQuad ) )
    {
        glDeleteBuffers ( 1, &mScreenQuad );
        mScreenQuad = 0;
    }
    if ( glIsVertexArray ( mVAO ) )
    {
        glDeleteVertexArrays ( 1, &mVAO );
        mVAO = 0;
    }
    if ( glIsProgram ( mProgram ) )
    {
        glUseProgram ( 0 );
        glDeleteProgram ( mProgram );
        mProgram = 0;
    }
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
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glUseProgram ( mProgram );
    glBindVertexArray ( mVAO );
    glDisable ( GL_DEPTH_TEST );
    mWindow.Draw();
    glBindTexture ( GL_TEXTURE_2D, mScreenTexture );
    glTexImage2D ( GL_TEXTURE_2D,
                   0,
                   GL_RGBA,
                   static_cast<GLsizei> ( mWindow.GetWidth() ),
                   static_cast<GLsizei> ( mWindow.GetHeight() ),
                   0,
                   GL_BGRA,
                   GL_UNSIGNED_INT_8_8_8_8_REV,
                   mWindow.GetPixels() );
    glDrawArrays ( GL_TRIANGLE_FAN, 0, 4 );
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
        break;
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
        lresult = window_ptr->OnMouseButtonDown ( 1, GET_X_LPARAM ( lParam ), GET_Y_LPARAM ( lParam ) );
        break;
    case WM_LBUTTONUP:
        lresult = window_ptr->OnMouseButtonUp ( 1, GET_X_LPARAM ( lParam ), GET_Y_LPARAM ( lParam ) );
        break;
#if 0
    case WM_SETCURSOR:
        if ( LOWORD ( lParam ) == HTCLIENT )
        {
            SetCursor ( NULL );
            return 0;
        }
        else
        {
            return DefWindowProc ( hwnd, uMsg, wParam, lParam );
        }
        break;
#endif
    default:
        lresult = DefWindowProc ( hwnd, uMsg, wParam, lParam );
    }
    return lresult;
}

LRESULT Window::OnSize ( WPARAM type, WORD newwidth, WORD newheight )
{
    LONG width = static_cast<LONG> ( newwidth );
    LONG height = static_cast<LONG> ( newheight );
    if ( height == 0 )
    {
        height = 1;
    }
    if ( width == 0 )
    {
        width = 1;
    }
    glViewport ( 0, 0, width, height );
    OPENGL_CHECK_ERROR;
    mWindow.ResizeViewport ( static_cast<size_t> ( newwidth ), static_cast<size_t> ( newheight ) );
    std::cout << "Width: " << mWindow.GetWidth() << std::endl;
    std::cout << "Height: " << mWindow.GetHeight() << std::endl;
    std::cout << "Stride: " << mWindow.GetStride() << std::endl;
    glTexImage2D ( GL_TEXTURE_2D,
                   0,
                   GL_RGBA,
                   width,
                   height,
                   0,
                   GL_BGRA,
                   GL_UNSIGNED_INT_8_8_8_8_REV,
                   mWindow.GetPixels() );
    OPENGL_CHECK_ERROR;
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

LRESULT Window::OnMouseButtonDown ( uint8_t button, int32_t x, int32_t y )
{
    return 0;
}

LRESULT Window::OnMouseButtonUp ( uint8_t button, int32_t x, int32_t y )
{
    return 0;
}

/** Convert a WinMain command line (lpCmdLine) into a regular argc,argv pair.
 * @param aCmdLine Windows API WinMain format command line.
 * @return tuple containing a vector of char* (std::get<0>) and a string
 * containing the argument strings separated each by a null character(std::get<1>).
 * @note The string part is meant only as managed storage for the char* vector,
 * as those point into the string memory, so there should not be a real reason
 * to read it directly.
*/
static std::tuple<std::vector<char*>, const std::string> GetArgs ( char* aCmdLine )
{
    /*Match any whitespace OR any whitespace followed by eol OR eol.*/
    std::regex whitespace{"\\s+|\\s+$|$"};
    std::tuple<std::vector<char*>, std::string>
    /*Replace all matches with a single plain old space character.
        Note: using "\0" instead of " " would be great, but "\0"
        gets converted as per the spec into the empty string rather
        than a size 1 string containing the null character.
        And no, "\0\0" does not work, std::string("\0\0",2) does not either.*/
    result{std::vector<char*>{}, std::regex_replace ( aCmdLine, whitespace, " " ) };

    size_t count{0};
    for ( size_t i = 0; i < std::get<1> ( result ).size(); ++i )
    {
        if ( std::get<1> ( result ) [i] == ' ' )
        {
            std::get<1> ( result ) [i] = '\0';
            ++count;
        }
    }

    if ( count )
    {
        std::get<0> ( result ).reserve ( count );
        std::get<0> ( result ).emplace_back ( std::get<1> ( result ).data() );
        for ( uint32_t i = 0; i < std::get<1> ( result ).size() - 1; ++i )
        {
            if ( std::get<1> ( result ) [i] == '\0' )
            {
                std::get<0> ( result ).emplace_back ( std::get<1> ( result ).data() + ( i + 1 ) );
            }
        }
    }
    return result;
}

int main ( int argc, char *argv[] )
{
    AeonGUI::Initialize ( argc, argv );
    if ( argc <= 1 )
    {
        std::cout << "You must provide the path to a SVG document" << std::endl;
        return -1;
    }

    MSG msg;
    try
    {
        // This Context allows the window variable to be destroyed before the call to AeonGUI::Finalize
        {
            Window window ( GetModuleHandle ( NULL ), argv[1], 800, 600 );
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
        }
        assert ( msg.message == WM_QUIT );
    }
    catch ( std::runtime_error& e )
    {
        std::cerr << e.what() << std::endl;
    }
    AeonGUI::Finalize();
    return static_cast<int> ( msg.wParam );
}

int WINAPI WinMain ( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
    auto args = GetArgs ( lpCmdLine );
    return main ( static_cast<int> ( std::get<0> ( args ).size() ), std::get<0> ( args ).data() );
}
