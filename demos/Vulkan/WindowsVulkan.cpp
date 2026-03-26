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

#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <windowsx.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdint>
#include <string>
#include <regex>
#include <tuple>
#include <vector>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "VulkanRenderer.h"

// ─────────────────────────────────────────────────────────────────────────────
class VulkanWindow
{
public:
    VulkanWindow ( HINSTANCE hInstance, const char* filename, LONG aWidth, LONG aHeight );
    ~VulkanWindow();
    void RenderLoop();
    static LRESULT CALLBACK WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
    static void Register ( HINSTANCE hInstance );
private:
    void OnResize ( WORD newWidth, WORD newHeight );
    static ATOM atom;

    HWND                    mHWnd{};
    uint32_t                mWidth;
    uint32_t                mHeight;
    std::string             mVertPath;
    std::string             mFragPath;
    AeonGUI::DOM::Window    mAeonWindow;
    VulkanRenderer          mRenderer;
};

ATOM VulkanWindow::atom = 0;

void VulkanWindow::Register ( HINSTANCE hInstance )
{
    WNDCLASSEX wcex{};
    wcex.cbSize        = sizeof ( WNDCLASSEX );
    wcex.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wcex.lpfnWndProc   = VulkanWindow::WindowProc;
    wcex.cbWndExtra    = sizeof ( VulkanWindow* );
    wcex.hInstance     = hInstance;
    wcex.hIcon         = LoadIcon ( NULL, IDI_WINLOGO );
    wcex.hCursor       = LoadCursor ( NULL, IDC_ARROW );
    wcex.lpszClassName = "AeonGUIVulkan";
    VulkanWindow::atom = RegisterClassEx ( &wcex );
}

VulkanWindow::VulkanWindow ( HINSTANCE hInstance, const char* filename, LONG aWidth, LONG aHeight )
    : mWidth ( static_cast<uint32_t> ( aWidth ) )
    , mHeight ( static_cast<uint32_t> ( aHeight ) )
    , mAeonWindow ( mWidth, mHeight )
{
    if ( filename )
    {
        mAeonWindow.location() = filename;
    }

    if ( atom == 0 )
    {
        Register ( hInstance );
    }

    RECT rect = { 0, 0, aWidth, aHeight };
    AdjustWindowRectEx ( &rect,
                         WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, FALSE,
                         WS_EX_APPWINDOW | WS_EX_WINDOWEDGE );

    mHWnd = CreateWindowEx (
                WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
                "AeonGUIVulkan", "AeonGUI - Vulkan",
                WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                CW_USEDEFAULT, CW_USEDEFAULT,
                rect.right - rect.left, rect.bottom - rect.top,
                NULL, NULL, hInstance, this );
    SetWindowLongPtr ( mHWnd, 0, reinterpret_cast<LONG_PTR> ( this ) );

    // Determine shader path (next to executable)
    char exePath[MAX_PATH];
    GetModuleFileNameA ( NULL, exePath, MAX_PATH );
    std::string dir ( exePath );
    dir = dir.substr ( 0, dir.rfind ( '\\' ) + 1 );
    mVertPath = dir + "screen.vert.spv";
    mFragPath = dir + "screen.frag.spv";

    // Vulkan setup
    mRenderer.CreateInstance ( { VK_KHR_WIN32_SURFACE_EXTENSION_NAME } );

    VkWin32SurfaceCreateInfoKHR surfaceInfo{};
    surfaceInfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.hinstance = hInstance;
    surfaceInfo.hwnd      = mHWnd;

    VkSurfaceKHR surface;
    VK_CHECK ( vkCreateWin32SurfaceKHR ( mRenderer.GetInstance(), &surfaceInfo, nullptr, &surface ) );
    mRenderer.SetSurface ( surface );

    mRenderer.Initialize ( mWidth, mHeight, mVertPath, mFragPath,
                           static_cast<uint32_t> ( mAeonWindow.GetWidth() ),
                           static_cast<uint32_t> ( mAeonWindow.GetHeight() ) );

    ShowWindow ( mHWnd, SW_SHOW );
}

VulkanWindow::~VulkanWindow()
{
    mRenderer.Cleanup();
    DestroyWindow ( mHWnd );
}

void VulkanWindow::OnResize ( WORD newWidth, WORD newHeight )
{
    mWidth  = ( newWidth  == 0 ) ? 1 : newWidth;
    mHeight = ( newHeight == 0 ) ? 1 : newHeight;
    mRenderer.RecreateSwapchain ( mWidth, mHeight, mAeonWindow, mVertPath, mFragPath );
}

void VulkanWindow::RenderLoop()
{
    mRenderer.DrawFrame ( mAeonWindow );
}

LRESULT CALLBACK VulkanWindow::WindowProc ( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    VulkanWindow* self = reinterpret_cast<VulkanWindow*> ( GetWindowLongPtr ( hwnd, 0 ) );
    switch ( uMsg )
    {
    case WM_CLOSE:
        PostQuitMessage ( 0 );
        return 0;
    case WM_SIZE:
        if ( self )
        {
            self->OnResize ( LOWORD ( lParam ), HIWORD ( lParam ) );
        }
        return 0;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        BeginPaint ( hwnd, &ps );
        EndPaint ( hwnd, &ps );
        return 0;
    }
    case WM_MOUSEMOVE:
        if ( self )
        {
            self->mAeonWindow.HandleMouseMove (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ) );
        }
        return 0;
    case WM_LBUTTONDOWN:
        if ( self )
        {
            self->mAeonWindow.HandleMouseDown (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 0 );
        }
        return 0;
    case WM_LBUTTONUP:
        if ( self )
        {
            self->mAeonWindow.HandleMouseUp (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 0 );
        }
        return 0;
    case WM_MBUTTONDOWN:
        if ( self )
        {
            self->mAeonWindow.HandleMouseDown (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 1 );
        }
        return 0;
    case WM_MBUTTONUP:
        if ( self )
        {
            self->mAeonWindow.HandleMouseUp (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 1 );
        }
        return 0;
    case WM_RBUTTONDOWN:
        if ( self )
        {
            self->mAeonWindow.HandleMouseDown (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 2 );
        }
        return 0;
    case WM_RBUTTONUP:
        if ( self )
        {
            self->mAeonWindow.HandleMouseUp (
                static_cast<double> ( GET_X_LPARAM ( lParam ) ),
                static_cast<double> ( GET_Y_LPARAM ( lParam ) ), 2 );
        }
        return 0;
    case WM_MOUSEWHEEL:
        if ( self )
        {
            POINT pt = { GET_X_LPARAM ( lParam ), GET_Y_LPARAM ( lParam ) };
            ScreenToClient ( hwnd, &pt );
            self->mAeonWindow.HandleWheel (
                static_cast<double> ( pt.x ),
                static_cast<double> ( pt.y ),
                0.0, static_cast<double> ( -GET_WHEEL_DELTA_WPARAM ( wParam ) ) );
        }
        return 0;
    default:
        return DefWindowProc ( hwnd, uMsg, wParam, lParam );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int main ( int argc, char* argv[] )
{
    AeonGUI::Initialize ( argc, argv );
    {
        VulkanWindow window ( GetModuleHandle ( NULL ),
                              ( argc > 1 ) ? argv[1] : nullptr,
                              800, 600 );
        MSG msg{};
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
    AeonGUI::Finalize();
    return 0;
}

int WINAPI WinMain ( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
    return main ( __argc, __argv );
}
