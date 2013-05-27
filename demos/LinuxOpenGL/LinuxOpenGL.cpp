#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysymdef.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include "glxext.h"
#include "glext.h"


static PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = NULL;

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

static uint32_t GetScancode ( KeySym keysym )
{
    switch ( keysym )
    {
    case XK_W:
    case XK_w:
        return 0x11;
    case XK_A:
    case XK_a:
        return 0x1e;
    case XK_S:
    case XK_s:
        return 0x1f;
    case XK_D:
    case XK_d:
        return 0x20;
#if 0
        //Remove the key as it is included into the switch statement.
        KEY_ESC      = 0x01,
        KEY_1        = 0x02,
        KEY_2        = 0x03,
        KEY_3        = 0x04,
        KEY_4        = 0x05,
        KEY_5        = 0x06,
        KEY_6        = 0x07,
        KEY_7        = 0x08,
        KEY_8        = 0x09,
        KEY_9        = 0x0A,
        KEY_0        = 0x0B,
        KEY_DASH     = 0x0C,
        KEY_EQUAL    = 0x0D,
        KEY_BKSP     = 0x0E,

        KEY_TAB      = 0x0F,
        KEY_Q        = 0x10,
        KEY_E        = 0x12,
        KEY_R        = 0x13,
        KEY_T        = 0x14,
        KEY_Y        = 0x15,
        KEY_U        = 0x16,
        KEY_I        = 0x17,
        KEY_O        = 0x18,
        KEY_P        = 0x19,
        KEY_LBRACKET = 0x1A,
        KEY_RBRACKET = 0x1B,
        KEY_ENTER    = 0x1C,

        KEY_CTRL     = 0x1D,
        KEY_F        = 0x21,
        KEY_G        = 0x22,
        KEY_H        = 0x23,
        KEY_J        = 0x24,
        KEY_K        = 0x25,
        KEY_L        = 0x26,
        KEY_SEMICOLON = 0x27,
        KEY_SQUOTE   = 0x28,
        KEY_TILDE    = 0x29,

        KEY_LSHIFT   = 0x2A,
        KEY_BACKSLASH = 0x2B,
        KEY_Z        = 0x2C,
        KEY_X        = 0x2D,
        KEY_C        = 0x2E,
        KEY_V        = 0x2F,
        KEY_B        = 0x30,
        KEY_N        = 0x31,
        KEY_M        = 0x32,
        KEY_COMMA    = 0x33,
        KEY_PERIOD   = 0x34,
        KEY_SLASH    = 0x35,
        KEY_RSHIFT   = 0x36,

        KEY_PRTSCN   = 0x37,
        KEY_ALT      = 0x38,
        KEY_SPACE    = 0x39,
        KEY_CAPS     = 0x3A,

        KEY_F1       = 0x3B,
        KEY_F2       = 0x3C,
        KEY_F3       = 0x3D,
        KEY_F4       = 0x3E,
        KEY_F5       = 0x3F,
        KEY_F6       = 0x40,
        KEY_F7       = 0x41,
        KEY_F8       = 0x42,
        KEY_F9       = 0x43,
        KEY_F10      = 0x44,
        KEY_NUM      = 0x45,
        KEY_SCROLL   = 0x46,

        KEY_HOME     = 0x47,
        KEY_UP       = 0x48,
        KEY_PGUP     = 0x49,
        KEY_MINUS    = 0x4A,
        KEY_LEFT     = 0x4B,
        KEY_CENTRE   = 0x4C,
        KEY_RIGHT    = 0x4D,
        KEY_PLUS     = 0x4E,
        KEY_END      = 0x4F,
        KEY_DOWN     = 0x50,
        KEY_PGDN     = 0x51,
        KEY_INS      = 0x52,
        KEY_DEL      = 0x53
#endif
    }
    return keysym;
}

class GLWindow
{
public:
    GLWindow();
    ~GLWindow();
    bool Create ( Display* dpy );
    void Destroy();
private:
    Display* display;
    GLXContext ctx;
    Colormap cmap;
    Window window;
};

GLWindow::GLWindow() :
    display ( NULL ),
    ctx ( 0 ),
    cmap ( 0 ),
    window ( 0 )
{}

bool GLWindow::Create ( Display* dpy )
{
    assert ( dpy != NULL );
    display = dpy;
    // Get a matching FB config
    static int visual_attribs[] =
    {
        GLX_X_RENDERABLE    , True,
        GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
        GLX_RENDER_TYPE     , GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
        GLX_RED_SIZE        , 8,
        GLX_GREEN_SIZE      , 8,
        GLX_BLUE_SIZE       , 8,
        GLX_ALPHA_SIZE      , 8,
        GLX_DEPTH_SIZE      , 24,
        GLX_STENCIL_SIZE    , 8,
        GLX_DOUBLEBUFFER    , True,
        //GLX_SAMPLE_BUFFERS  , 1,
        //GLX_SAMPLES         , 4,
        None
    };

    printf ( "Getting matching framebuffer configs" );
    int fbcount;
    GLXFBConfig *fbc = glXChooseFBConfig ( display, DefaultScreen ( display ),
                                           visual_attribs, &fbcount );
    if ( !fbc )
    {
        printf ( "Failed to retrieve a framebuffer config" );
        return false;
    }
    printf ( "Found %d matching FB configs.", fbcount );

    // Pick the FB config/visual with the most samples per pixel
    printf ( "Getting XVisualInfos" );
    int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

    int i;
    for ( i = 0; i < fbcount; i++ )
    {
        XVisualInfo *vi = glXGetVisualFromFBConfig ( display, fbc[i] );
        if ( vi )
        {
            int samp_buf, samples;
            glXGetFBConfigAttrib ( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
            glXGetFBConfigAttrib ( display, fbc[i], GLX_SAMPLES       , &samples  );

            printf ( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
                                 " SAMPLES = %d",
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
    printf ( "Chosen visual ID = 0x%x", static_cast<unsigned int> ( vi->visualid ) );

    printf ( "Creating colormap" );
    XSetWindowAttributes swa;

    swa.colormap = cmap = XCreateColormap ( display,
                                            RootWindow ( display, vi->screen ),
                                            vi->visual, AllocNone );
    swa.background_pixmap = None ;
    swa.border_pixel      = 0;
    swa.event_mask        = StructureNotifyMask;

    printf ( "Creating window" );
    window = XCreateWindow ( display, RootWindow ( display, vi->screen ),
                             0, 0, 800, 600, 0, vi->depth, InputOutput,
                             vi->visual,
                             CWBorderPixel | CWColormap | CWEventMask, &swa );
    if ( !window )
    {
        printf ( "Failed to create window." );
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
                   ResizeRedirectMask );

    Atom wm_delete_window = XInternAtom ( display, "WM_DELETE_WINDOW", 0 );
    XSetWMProtocols ( display, window, &wm_delete_window, 1 );

    printf ( "Mapping window" );
    XMapWindow ( display, window );

    if ( glXCreateContextAttribsARB )
    {
        int context_attribs[] =
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
            GLX_CONTEXT_MINOR_VERSION_ARB, 2,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
            //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
            None
        };

        printf ( "Creating context" );
        ctx = glXCreateContextAttribsARB ( display, bestFbc, 0,
                                           True, context_attribs );
        XSync ( display, False );
        if ( ctx != NULL )
        {
            printf ( "Created GL %d.%d context", context_attribs[1], context_attribs[3] );
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
        printf ( "Indirect GLX rendering context obtained" );
    }
    else
    {
        printf ( "Direct GLX rendering context obtained" );
    }

    printf ( "Making context current" );
    glXMakeCurrent ( display, window, ctx );

    bool running = true;
    XEvent xEvent;

    glViewport( 0, 0, 800, 600 );

    timespec current_time;
    clock_gettime ( CLOCK_REALTIME, &current_time );
    timespec last_time = current_time;
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
            case ResizeRequest:
                glViewport(0, 0, xEvent.xresizerequest.width, xEvent.xresizerequest.height );
                break;
            case ClientMessage:
                if ( static_cast<Atom> ( xEvent.xclient.data.l[0] ) == wm_delete_window )
                {
                    running = false;
                }
                break;
            default:
                printf ( "Received Event Type: %d", xEvent.type );
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
    GLWindow glWindow;
    int glx_major, glx_minor;

    Display* display = XOpenDisplay ( 0 );
    if ( !display )
    {
        printf ( "Failed to open X display" );
        return EXIT_FAILURE;
    }

    // FBConfigs were added in GLX version 1.3.
    if ( !glXQueryVersion ( display, &glx_major, &glx_minor ) ||
         ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
    {
        printf ( "Invalid GLX version %d.%d", glx_major, glx_minor );
        XCloseDisplay ( display );
        return EXIT_FAILURE;
    }
    printf ( "GLX Version: %d.%d", glx_major, glx_minor );

    // Get the default screen's GLX extension list
    const char *glxExts = glXQueryExtensionsString ( display,
                          DefaultScreen ( display ) );

    // Check for the GLX_ARB_create_context extension string and the function.
    if ( !isExtensionSupported ( glxExts, "GLX_ARB_create_context" ) )
    {
        printf ( "GLX_ARB_create_context not supported" );
        XCloseDisplay ( display );
        return EXIT_FAILURE;
    }

    // NOTE: It is not necessary to create or make current to a context before
    // calling glXGetProcAddressARB
    glXCreateContextAttribsARB = ( PFNGLXCREATECONTEXTATTRIBSARBPROC )
                                 glXGetProcAddressARB ( ( const GLubyte * ) "glXCreateContextAttribsARB" );
    if ( glXCreateContextAttribsARB == NULL )
    {
        printf ( "Pointer for glXCreateContextAttribsARB is NULL" );
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
    return EXIT_SUCCESS;
}
