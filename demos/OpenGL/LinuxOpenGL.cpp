/*
Copyright (C) 2013,2019-2021,2023,2025 Rodrigo Jose Hernandez Cordoba

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
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include "aeongui/AeonGUI.h"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "Common.h"

class GLWindow
{
public:
    GLWindow ( char* aFilename );
    ~GLWindow();
    bool Create ( Display* dpy );
    void Destroy();
private:
    Display* display;
    GLXContext ctx;
    Colormap cmap;
    Window window;
    uint32_t mWidth;
    uint32_t mHeight;
    GLuint mScreenQuad{};
    GLuint mScreenTexture{};
    AeonGUI::DOM::Document mDocument;
    AeonGUI::DOM::Window mWindow;
};

GLWindow::GLWindow ( char* aFilename ) :
    display ( NULL ),
    ctx ( 0 ),
    cmap ( 0 ),
    window ( 0 ),
    mWidth ( 800 ),
    mHeight ( 600 ),
    mDocument ( aFilename ? aFilename : "" ),
    mWindow{&mDocument, static_cast<uint32_t> ( mWidth ), static_cast<uint32_t> ( mHeight ) }
{}

bool GLWindow::Create ( Display* dpy )
{
    assert ( dpy != NULL );
    display = dpy;
    // Get a matching FB config
    static int visual_attribs[] =
    {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_STENCIL_SIZE, 8,
        GLX_DOUBLEBUFFER, True,
        //GLX_SAMPLE_BUFFERS  , 1,
        //GLX_SAMPLES         , 4,
        None
    };

    printf ( "Getting matching framebuffer configs\n" );
    int fbcount;
    GLXFBConfig *fbc = glXChooseFBConfig ( display, DefaultScreen ( display ),
                                           visual_attribs, &fbcount );
    if ( !fbc )
    {
        printf ( "Failed to retrieve a framebuffer config\n" );
        return false;
    }
    printf ( "Found %d matching FB configs.\n", fbcount );

    // Pick the FB config/visual with the most samples per pixel
    printf ( "Getting XVisualInfos\n" );
    int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

    int i;
    for ( i = 0; i < fbcount; i++ )
    {
        XVisualInfo *vi = glXGetVisualFromFBConfig ( display, fbc[i] );
        if ( vi )
        {
            int samp_buf, samples;
            glXGetFBConfigAttrib ( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
            glXGetFBConfigAttrib ( display, fbc[i], GLX_SAMPLES, &samples  );

            printf ( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
                     " SAMPLES = %d\n",
                     i, static_cast<unsigned int> ( vi->visualid ), samp_buf, samples );

            if ( best_fbc < 0 || ( samp_buf && samples > best_num_samp ) )
            {
                best_fbc = i, best_num_samp = samples;
            }
            if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
            {
                worst_fbc = i, worst_num_samp = samples;
            }
        }
        XFree ( vi );
    }

    GLXFBConfig bestFbc = fbc[ best_fbc ];

    // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
    XFree ( fbc );

    // Get a visual
    XVisualInfo *vi = glXGetVisualFromFBConfig ( display, bestFbc );
    printf ( "Chosen visual ID = 0x%x\n", static_cast<unsigned int> ( vi->visualid ) );

    printf ( "Creating colormap\n" );
    XSetWindowAttributes swa;

    swa.colormap = cmap = XCreateColormap ( display,
                                            RootWindow ( display, vi->screen ),
                                            vi->visual, AllocNone );
    swa.background_pixmap = None ;
    swa.border_pixel      = 0;
    swa.event_mask        = KeyPressMask |
                            KeyReleaseMask |
                            ButtonPressMask |
                            ButtonReleaseMask |
                            PointerMotionMask |
                            StructureNotifyMask;

    printf ( "Creating window\n" );
    window = XCreateWindow ( display, RootWindow ( display, vi->screen ),
                             0, 0, mWidth, mHeight, 0, vi->depth, InputOutput,
                             vi->visual,
                             CWBorderPixel | CWColormap | CWEventMask, &swa );
    if ( !window )
    {
        printf ( "Failed to create window.\n" );
        return false;
    }

    XStoreName ( display, window, "AeonGUI" );

    XSelectInput ( display, window,
                   KeyPressMask |
                   KeyReleaseMask |
                   ButtonPressMask |
                   ButtonReleaseMask |
                   PointerMotionMask |
                   StructureNotifyMask );

    Atom wm_delete_window = XInternAtom ( display, "WM_DELETE_WINDOW", 0 );
    XSetWMProtocols ( display, window, &wm_delete_window, 1 );

    printf ( "Mapping window\n" );
    XMapWindow ( display, window );
    printf ( "Creating context\n" );
    ctx = glXCreateContext ( display, vi, nullptr, True );
    // Done with the visual info data
    XFree ( vi );
    XSync ( display, False );
    if ( ctx == NULL )
    {
        return false;
    }
    // Verifying that context is a direct context
    if ( ! glXIsDirect ( display, ctx ) )
    {
        printf ( "Indirect GLX rendering context obtained\n" );
    }
    else
    {
        printf ( "Direct GLX rendering context obtained\n" );
    }

    printf ( "Making context current\n" );
    glXMakeCurrent ( display, window, ctx );
    printf ( "GL_VERSION: %s\n", glGetString ( GL_VERSION ) );

    bool running = true;
    XEvent xEvent;

    glClearColor ( 1, 1, 1, 1 );
    glViewport ( 0, 0, mWidth, mHeight );
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    OPENGL_CHECK_ERROR;
    glEnable ( GL_BLEND );
    OPENGL_CHECK_ERROR;
    glGenTextures ( 1, &mScreenTexture );
    OPENGL_CHECK_ERROR;
    glBindTexture ( GL_TEXTURE_2D, mScreenTexture );
    OPENGL_CHECK_ERROR;
    glTexImage2D ( GL_TEXTURE_2D,
                   0,
                   GL_RGBA,
                   mWidth,
                   mHeight,
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

    timespec current_time;
    clock_gettime ( CLOCK_REALTIME, &current_time );
    static timespec last_time = current_time;
    float delta;
    while ( running )
    {
        while ( ( XPending ( display ) > 0 ) && running )
        {
            XNextEvent ( display, &xEvent );
            switch ( xEvent.type )
            {
            case KeyPress:
                break;
            case KeyRelease:
                break;
            case ButtonPress:
                break;
            case ButtonRelease:
                break;
            case MotionNotify:
                break;
            case ConfigureNotify:
                mWidth = xEvent.xconfigure.width;
                mHeight = xEvent.xconfigure.height;
                glViewport ( 0, 0, mWidth, mHeight );
                OPENGL_CHECK_ERROR;
                mWindow.ResizeViewport ( static_cast<size_t> ( mWidth ), static_cast<size_t> ( mHeight ) );
                glTexImage2D ( GL_TEXTURE_2D,
                               0,
                               GL_RGBA,
                               mWidth,
                               mHeight,
                               0,
                               GL_BGRA,
                               GL_UNSIGNED_INT_8_8_8_8_REV,
                               mWindow.GetPixels() );
                OPENGL_CHECK_ERROR;
                break;
            case ClientMessage:
                if ( static_cast<Atom> ( xEvent.xclient.data.l[0] ) == wm_delete_window )
                {
                    running = false;
                }
                break;
            default:
                printf ( "Received Event Type: %d\n", xEvent.type );
            }
        }
        clock_gettime ( CLOCK_REALTIME, &current_time );
        delta = static_cast<float> ( current_time.tv_sec - last_time.tv_sec )   +
                static_cast<float> ( current_time.tv_nsec - last_time.tv_nsec ) * 1e-9;
        if ( delta > 1e-1 )
        {
            delta = 1.0f / 30.0f;
        }
        last_time = current_time;

        mWindow.Draw();

        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        OPENGL_CHECK_ERROR;
        glBindTexture ( GL_TEXTURE_2D, mScreenTexture );
        OPENGL_CHECK_ERROR;
        glTexSubImage2D ( GL_TEXTURE_2D,
                          0,
                          0,
                          0,
                          static_cast<GLsizei> ( mWindow.GetWidth() ),
                          static_cast<GLsizei> ( mWindow.GetHeight() ),
                          GL_BGRA,
                          GL_UNSIGNED_INT_8_8_8_8_REV,
                          mWindow.GetPixels() );
        OPENGL_CHECK_ERROR;
        glEnable ( GL_TEXTURE_2D );
        glBegin ( GL_TRIANGLE_FAN );
        {
            glTexCoord2f ( vertices[2], vertices[3] );
            glVertex2f ( vertices[0], vertices[1] );
            glTexCoord2f ( vertices[6], vertices[7] );
            glVertex2f ( vertices[4], vertices[5] );
            glTexCoord2f ( vertices[10], vertices[11] );
            glVertex2f ( vertices[8], vertices[9] );
            glTexCoord2f ( vertices[14], vertices[15] );
            glVertex2f ( vertices[12], vertices[13] );
        }
        glEnd();
        OPENGL_CHECK_ERROR;
        glXSwapBuffers ( display, window );
    }
    return true;
}

void GLWindow::Destroy()
{
    if ( display != NULL )
    {
        glXMakeCurrent ( display, 0, 0 );
        if ( ctx != 0 )
        {
            glXDestroyContext ( display, ctx );
            ctx = 0;
        }
        if ( window != 0 )
        {
            XDestroyWindow ( display, window );
            window = 0;
        }
        if ( cmap != 0 )
        {
            XFreeColormap ( display, cmap );
            cmap = 0;
        }
        display = NULL;
    }
}

GLWindow::~GLWindow()
{
    Destroy();
}


int main ( int argc, char ** argv )
{
    AeonGUI::Initialize ( argc, argv );
    {
        GLWindow glWindow ( ( argc > 1 ) ? argv[1] : nullptr );
        int glx_major, glx_minor;

        Display* display = XOpenDisplay ( 0 );
        if ( !display )
        {
            printf ( "Failed to open X display\n" );
            return EXIT_FAILURE;
        }

        // FBConfigs were added in GLX version 1.3.
        if ( !glXQueryVersion ( display, &glx_major, &glx_minor ) ||
             ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
        {
            printf ( "Invalid GLX version %d.%d\n", glx_major, glx_minor );
            XCloseDisplay ( display );
            return EXIT_FAILURE;
        }
        printf ( "GLX Version: %d.%d\n", glx_major, glx_minor );

        if ( !glWindow.Create ( display ) )
        {
            XCloseDisplay ( display );
            return EXIT_FAILURE;
        }
        glWindow.Destroy();
        XCloseDisplay ( display );
    }
    AeonGUI::Finalize();
    return EXIT_SUCCESS;
}
