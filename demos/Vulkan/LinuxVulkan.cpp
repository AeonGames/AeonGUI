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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>

#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "VulkanRenderer.h"

int main ( int argc, char** argv )
{
    AeonGUI::Initialize ( argc, argv );
    {
        uint32_t width  = 800;
        uint32_t height = 600;

        AeonGUI::DOM::Window aeonWindow ( width, height );
        if ( argc > 1 )
        {
            aeonWindow.location() = argv[1];
        }

        // ── X11 window creation ──────────────────────────────────────────
        Display* display = XOpenDisplay ( nullptr );
        if ( !display )
        {
            std::cerr << "Failed to open X display" << std::endl;
            return EXIT_FAILURE;
        }

        int screen = DefaultScreen ( display );
        ::Window xWindow = XCreateSimpleWindow ( display, RootWindow ( display, screen ),
                           0, 0, width, height, 1,
                           BlackPixel ( display, screen ),
                           WhitePixel ( display, screen ) );

        XStoreName ( display, xWindow, "AeonGUI \xe2\x80\x93 Vulkan" );
        XSelectInput ( display, xWindow,
                       KeyPressMask | KeyReleaseMask |
                       ButtonPressMask | ButtonReleaseMask |
                       PointerMotionMask | StructureNotifyMask );

        Atom wmDeleteWindow = XInternAtom ( display, "WM_DELETE_WINDOW", False );
        XSetWMProtocols ( display, xWindow, &wmDeleteWindow, 1 );
        XMapWindow ( display, xWindow );

        // ── Vulkan setup ─────────────────────────────────────────────────
        VulkanRenderer renderer;
        renderer.CreateInstance ( { VK_KHR_XLIB_SURFACE_EXTENSION_NAME } );

        VkXlibSurfaceCreateInfoKHR surfaceInfo{};
        surfaceInfo.sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        surfaceInfo.dpy    = display;
        surfaceInfo.window = xWindow;

        VkSurfaceKHR surface;
        VK_CHECK ( vkCreateXlibSurfaceKHR ( renderer.GetInstance(), &surfaceInfo, nullptr, &surface ) );
        renderer.SetSurface ( surface );

        std::string binDir = argv[0];
        binDir = binDir.substr ( 0, binDir.rfind ( '/' ) + 1 );
        std::string vertPath = binDir + "screen.vert.spv";
        std::string fragPath = binDir + "screen.frag.spv";

        renderer.Initialize ( width, height, vertPath, fragPath,
                              static_cast<uint32_t> ( aeonWindow.GetWidth() ),
                              static_cast<uint32_t> ( aeonWindow.GetHeight() ) );

        // ── Event loop ───────────────────────────────────────────────────
        bool running = true;
        XEvent xEvent;
        while ( running )
        {
            while ( XPending ( display ) > 0 )
            {
                XNextEvent ( display, &xEvent );
                switch ( xEvent.type )
                {
                case ConfigureNotify:
                    width  = static_cast<uint32_t> ( xEvent.xconfigure.width );
                    height = static_cast<uint32_t> ( xEvent.xconfigure.height );
                    renderer.RecreateSwapchain ( width, height, aeonWindow, vertPath, fragPath );
                    break;
                case ClientMessage:
                    if ( static_cast<Atom> ( xEvent.xclient.data.l[0] ) == wmDeleteWindow )
                    {
                        running = false;
                    }
                    break;
                case KeyPress:
                case KeyRelease:
                    break;
                case ButtonPress:
                    if ( xEvent.xbutton.button == Button1 )
                    {
                        aeonWindow.HandleMouseDown ( static_cast<double> ( xEvent.xbutton.x ),
                                                     static_cast<double> ( xEvent.xbutton.y ), 0 );
                    }
                    else if ( xEvent.xbutton.button == Button2 )
                    {
                        aeonWindow.HandleMouseDown ( static_cast<double> ( xEvent.xbutton.x ),
                                                     static_cast<double> ( xEvent.xbutton.y ), 1 );
                    }
                    else if ( xEvent.xbutton.button == Button3 )
                    {
                        aeonWindow.HandleMouseDown ( static_cast<double> ( xEvent.xbutton.x ),
                                                     static_cast<double> ( xEvent.xbutton.y ), 2 );
                    }
                    else if ( xEvent.xbutton.button == Button4 )
                    {
                        aeonWindow.HandleWheel ( static_cast<double> ( xEvent.xbutton.x ),
                                                 static_cast<double> ( xEvent.xbutton.y ),
                                                 0.0, -120.0 );
                    }
                    else if ( xEvent.xbutton.button == Button5 )
                    {
                        aeonWindow.HandleWheel ( static_cast<double> ( xEvent.xbutton.x ),
                                                 static_cast<double> ( xEvent.xbutton.y ),
                                                 0.0, 120.0 );
                    }
                    break;
                case ButtonRelease:
                    if ( xEvent.xbutton.button == Button1 )
                    {
                        aeonWindow.HandleMouseUp ( static_cast<double> ( xEvent.xbutton.x ),
                                                   static_cast<double> ( xEvent.xbutton.y ), 0 );
                    }
                    else if ( xEvent.xbutton.button == Button2 )
                    {
                        aeonWindow.HandleMouseUp ( static_cast<double> ( xEvent.xbutton.x ),
                                                   static_cast<double> ( xEvent.xbutton.y ), 1 );
                    }
                    else if ( xEvent.xbutton.button == Button3 )
                    {
                        aeonWindow.HandleMouseUp ( static_cast<double> ( xEvent.xbutton.x ),
                                                   static_cast<double> ( xEvent.xbutton.y ), 2 );
                    }
                    break;
                case MotionNotify:
                    aeonWindow.HandleMouseMove ( static_cast<double> ( xEvent.xmotion.x ),
                                                 static_cast<double> ( xEvent.xmotion.y ) );
                    break;
                default:
                    break;
                }
            }
            if ( running )
            {
                renderer.DrawFrame ( aeonWindow );
            }
        }

        renderer.Cleanup();
        XDestroyWindow ( display, xWindow );
        XCloseDisplay ( display );
    }
    AeonGUI::Finalize();
    return EXIT_SUCCESS;
}
