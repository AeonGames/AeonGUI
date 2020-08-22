/*
Copyright (C) 2013,2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Window.h"
#include "Common.h"

#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
    glFunction = ( glFunctionType ) glXGetProcAddress ( (const GLubyte*) #glFunction ); \
        if (glFunction == nullptr) \
        { \
            std::ostringstream stream; \
            stream << "OpenGL: Unable to load " #glFunction " function."; \
            throw std::runtime_error(stream.str().c_str()); \
        } \
    }

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
GLDEFINEFUNCTION ( PFNGLUNIFORM1IPROC, glUniform1i );
GLDEFINEFUNCTION ( PFNGLGETSHADERIVPROC, glGetShaderiv );
GLDEFINEFUNCTION ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
GLDEFINEFUNCTION ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
GLDEFINEFUNCTION ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );
GLDEFINEFUNCTION ( PFNGLXCREATECONTEXTATTRIBSARBPROC, glXCreateContextAttribsARB );

// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static bool isExtensionSupported ( const char *extList, const char *extension )
{
    const char *start;
    const char *where, *terminator;

    /* Extension names should not have spaces. */
    where = strchr ( extension, ' ' );
    if ( where || *extension == '\0' )
    {
        return false;
    }

    /* It takes a bit of care to be fool-proof about parsing the
       OpenGL extensions string. Don't be fooled by sub-strings,
       etc. */
    for ( start = extList; ; )
    {
        where = strstr ( start, extension );

        if ( !where )
        {
            break;
        }

        terminator = where + strlen ( extension );

        if ( where == start || * ( where - 1 ) == ' ' )
            if ( *terminator == ' ' || *terminator == '\0' )
            {
                return true;
            }

        start = terminator;
    }

    return false;
}

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
    GLuint mProgram{};
    GLuint mVAO{};
    GLuint mScreenQuad{};
    GLuint mScreenTexture{};
    AeonGUI::Window mWindow;
};

GLWindow::GLWindow ( char* aFilename ) :
    display ( NULL ),
    ctx ( 0 ),
    cmap ( 0 ),
    window ( 0 ),
    mWidth ( 800 ),
    mHeight ( 600 ),
    mWindow{aFilename ? aFilename : "", static_cast<uint32_t> ( mWidth ), static_cast<uint32_t> ( mHeight ) }
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

    // Done with the visual info data
    XFree ( vi );

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

    if ( glXCreateContextAttribsARB )
    {
        int context_attribs[] =
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 5,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        printf ( "Creating context\n" );
        ctx = glXCreateContextAttribsARB ( display, bestFbc, 0,
                                           True, context_attribs );
        XSync ( display, False );
        if ( ctx != NULL )
        {
            printf ( "Created GL %d.%d context\n", context_attribs[1], context_attribs[3] );
        }
        else
        {
            return false;
        }
    }
    else
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
    GLGETPROCADDRESS ( PFNGLUNIFORM1IPROC, glUniform1i );
    GLGETPROCADDRESS ( PFNGLGETSHADERIVPROC, glGetShaderiv );
    GLGETPROCADDRESS ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
    GLGETPROCADDRESS ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
    GLGETPROCADDRESS ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );

    bool running = true;
    XEvent xEvent;

    glClearColor ( 1, 1, 1, 1 );
    glViewport ( 0, 0, mWidth, mHeight );
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
                std::cout << "Width: " << mWindow.GetWidth() << std::endl;
                std::cout << "Height: " << mWindow.GetHeight() << std::endl;
                std::cout << "Stride: " << mWindow.GetStride() << std::endl;
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

    // Get the default screen's GLX extension list
    const char *glxExts = glXQueryExtensionsString ( display,
                          DefaultScreen ( display ) );

    // Check for the GLX_ARB_create_context extension string and the function.
    if ( !isExtensionSupported ( glxExts, "GLX_ARB_create_context" ) )
    {
        printf ( "GLX_ARB_create_context not supported\n" );
        XCloseDisplay ( display );
        return EXIT_FAILURE;
    }

    // NOTE: It is not necessary to create or make current to a context before
    // calling glXGetProcAddressARB
    glXCreateContextAttribsARB = ( PFNGLXCREATECONTEXTATTRIBSARBPROC )
                                 glXGetProcAddressARB ( ( const GLubyte * ) "glXCreateContextAttribsARB" );
    if ( glXCreateContextAttribsARB == NULL )
    {
        printf ( "Pointer for glXCreateContextAttribsARB is NULL\n" );
        XCloseDisplay ( display );
        return EXIT_FAILURE;
    }

    if ( !glWindow.Create ( display ) )
    {
        XCloseDisplay ( display );
        return EXIT_FAILURE;
    }
    glWindow.Destroy();
    XCloseDisplay ( display );
    AeonGUI::Finalize();
    return EXIT_SUCCESS;
}
