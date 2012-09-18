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
#include <iostream>
#if 0
#include <sstream>
#include <fstream>
#endif
#include "SDL.h"
#include "SDL_opengl.h"
#include "SDL_syswm.h"
#include "logo.h"
#include "Vera.h"
#include "Widget.h"
#include "MainWindow.h"
#include "Static.h"
#include "OpenGLRenderer.h"
#include "glcommon.h"

void DrawAxes()
{
    glBegin ( GL_LINES );
    // RED
    glColor3f ( 1.0f, 0.0f, 0.0f );
    glVertex3f ( 0.0f, 0.0f, 0.0f );
    glVertex3f ( 1.0f, 0.0f, 0.0f );
    // GREEN
    glColor3f ( 0.0f, 1.0f, 0.0f );
    glVertex3f ( 0.0f, 0.0f, 0.0f );
    glVertex3f ( 0.0f, 1.0f, 0.0f );
    // BLUE
    glColor3f ( 0.0f, 0.0f, 1.0f );
    glVertex3f ( 0.0f, 0.0f, 0.0f );
    glVertex3f ( 0.0f, 0.0f, 1.0f );
    glEnd();
    glBegin ( GL_POINTS );
    glColor3f ( 1.0f, 0.0f, 0.0f );
    glVertex3f ( 1.0f, 0.0f, 0.0f );
    glColor3f ( 0.0f, 1.0f, 0.0f );
    glVertex3f ( 0.0f, 1.0f, 0.0f );
    glColor3f ( 0.0f, 0.0f, 1.0f );
    glVertex3f ( 0.0f, 0.0f, 1.0f );
    glEnd();
}

int main ( int argc, char *argv[] )
{
    std::wstring hello ( L"Hello World" );
    fprintf ( stdout, "Demo Starting\n" );
    bool bRunning = true;
    SDL_Event event;
    //SDL_Surface *screen=NULL;
    SDL_SysWMinfo SysWMinfo;
    if ( SDL_Init ( SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE ) < 0 )
    {
        fprintf ( stderr, "Couldn't initialize SDL: %s\n", SDL_GetError() );
        return ( 1 );
    }

    fprintf ( stdout, "Available Video Modes\n" );
    SDL_Rect **modes;
    modes = SDL_ListModes ( NULL, SDL_OPENGL );
    if ( modes == NULL )
    {
        fprintf ( stdout, "No modes available!\n" );
    }
    if ( modes == ( SDL_Rect ** ) - 1 )
    {
        fprintf ( stdout, "All resolutions available.\n" );
    }
    else
    {
        for ( int i = 0; modes[i]; ++i )
        {
            fprintf ( stdout, "\t%d x %d\n", modes[i]->w, modes[i]->h );
        }
    }

    SDL_GL_SetAttribute ( SDL_GL_RED_SIZE, 8 );
    SDL_GL_SetAttribute ( SDL_GL_GREEN_SIZE, 8 );
    SDL_GL_SetAttribute ( SDL_GL_BLUE_SIZE, 8 );
    SDL_GL_SetAttribute ( SDL_GL_ALPHA_SIZE, 8 );
    SDL_GL_SetAttribute ( SDL_GL_DEPTH_SIZE, 16 );
    //SDL_GL_SetAttribute( SDL_GL_STENCIL_SIZE, 16);
    SDL_GL_SetAttribute ( SDL_GL_DOUBLEBUFFER, 1 );
    if ( SDL_SetVideoMode ( 640, 480, 0, SDL_OPENGL ) == 0 )
    {
        fprintf ( stderr, "Couldn't set video mode: %s\n", SDL_GetError() );
        return ( 1 );
    }
    SDL_WM_SetCaption ( "AeonGUI 1.0 SDL OpenGL Demo", "" );
    SDL_EnableKeyRepeat ( SDL_DEFAULT_REPEAT_DELAY, SDL_DEFAULT_REPEAT_INTERVAL );
    AeonGUI::OpenGLRenderer renderer;
    AeonGUI::Color color = 0xFFFFFFFF;
    const SDL_VideoInfo* video_info = SDL_GetVideoInfo();
    // video_info contains window resolution, but thats ok since the SDL window's size is static.
    if ( !renderer.Initialize ( video_info->current_w, video_info->current_h ) )
    {
        return 1;
    }

    AeonGUI::Image* image = new AeonGUI::Image ( logo_name, logo_width, logo_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, logo_data );

    AeonGUI::Font* font = new AeonGUI::Font ( Vera.data, Vera.size );

    renderer.SetFont ( font );

    AeonGUI::MainWindow window;
    window.SetCaption ( hello );

    printf ( "OpenGL Version %s\n", glGetString ( GL_VERSION ) );
    printf ( "OpenGL Vendor %s\n", glGetString ( GL_VENDOR ) );
    printf ( "Extensions %s\n", glGetString ( GL_EXTENSIONS ) );

    glShadeModel ( GL_SMOOTH );
    glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
    glClearDepth ( 1.0f );
    glHint ( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    SDL_VERSION ( &SysWMinfo.version );
    SDL_GetWMInfo ( &SysWMinfo );
    SDL_EnableUNICODE ( 1 );
    //glViewport(0,0,640,480);
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
    //glPointSize(16);
#if 1
    float rotation = 0.0f;
    Uint32 lasttime = SDL_GetTicks();
    Uint32 thistime;
    Uint32 delta = 0;
#endif
    fprintf ( stdout, "Entering Main Loop.\n" );
    while ( bRunning )
    {
        while ( SDL_PollEvent ( &event ) )
        {
            switch ( event.type )
            {
            case SDL_QUIT:
                bRunning = false;
                std::cout << "Quit Event" << std::endl;
                break;
            case SDL_KEYDOWN:
                if ( event.key.keysym.sym == SDLK_ESCAPE )
                {
                    bRunning = false;
                    std::cout << "ESC Pressed" << std::endl;
                }
                window.KeyDown ( event.key.keysym.unicode );
                break;
            case SDL_KEYUP:
                window.KeyUp ( event.key.keysym.unicode );
                break;
            case SDL_MOUSEMOTION:
                window.MouseMove ( event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel );
                break;
            case SDL_MOUSEBUTTONDOWN:
                if ( window.IsPointInside ( event.button.x, event.button.y ) )
                {
                    window.MouseButtonDown ( event.button.button, event.button.x, event.button.y );
                }
                break;
            case SDL_MOUSEBUTTONUP:
                if ( window.IsPointInside ( event.button.x, event.button.y ) )
                {
                    window.MouseButtonUp ( event.button.button, event.button.x, event.button.y );
                }
                break;
            }
        }
        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glLoadIdentity();
#if 1
        // Move to
        glTranslatef ( 0.0f, 0.0f, 7.0f );
        // Rotate around current pos
        thistime = SDL_GetTicks();
        delta = ( thistime - lasttime );
        lasttime = thistime;
        glRotatef ( rotation += ( 0.05f * delta ), 1.0f, 1.0f, 1.0f );
        if ( rotation >= 360.0f )
        {
            rotation = 0;
        }
        DrawCube ( 1.0f );
        //DrawAxes();
#endif
        renderer.BeginRender();
        window.Render ( &renderer );
        renderer.DrawImage ( color, 512, 352, image );
        //renderer.DrawImage(color,512+64,352+64,image);
        //renderer.DrawImage(color,-64,-64,image);
        renderer.EndRender();
        SDL_GL_SwapBuffers();
    }
//  renderer.ReleaseImage(image);
    if ( font != NULL )
    {
        delete font;
    }
    if ( image != NULL )
    {
        delete image;
    }
    renderer.Finalize();
    SDL_Quit();
#if 0
    fprintf ( stdout, "printf\n" );
    printf ( "printf\n" );
    std::cout << "Just making sure we get something in there" << std::endl;
    std::ofstream filestr;
    filestr.open ( "test.txt" );
    filestr << sout.str();
    filestr.close();
    std::cout.rdbuf ( backup );
#endif
    return 0;
};
